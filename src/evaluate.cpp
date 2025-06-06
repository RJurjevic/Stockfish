/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2020 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>   // For std::memset
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <streambuf>
#include <vector>

#include "nnue/evaluate_nnue.h"

#include "bitboard.h"
#include "evaluate.h"
#include "material.h"
#include "misc.h"
#include "pawns.h"
#include "thread.h"
#include "uci.h"
#include "incbin/incbin.h"

using namespace std;

namespace Trace {

  enum Tracing { NO_TRACE, TRACE };

  enum Term { // The first 8 entries are reserved for PieceType
    MATERIAL = 8, IMBALANCE, MOBILITY, THREAT, PASSED, SPACE, WINNABLE, TOTAL, TERM_NB
  };

  Score scores[TERM_NB][COLOR_NB];

  double to_cp(Value v) { return double(v) / PawnValueEg; }

  void add(int idx, Color c, Score s) {
    scores[idx][c] = s;
  }

  void add(int idx, Score w, Score b = SCORE_ZERO) {
    scores[idx][WHITE] = w;
    scores[idx][BLACK] = b;
  }

  std::ostream& operator<<(std::ostream& os, Score s) {
    os << std::setw(5) << to_cp(mg_value(s)) << " "
       << std::setw(5) << to_cp(eg_value(s));
    return os;
  }

  std::ostream& operator<<(std::ostream& os, Term t) {

    if (t == MATERIAL || t == IMBALANCE || t == WINNABLE || t == TOTAL)
        os << " ----  ----"    << " | " << " ----  ----";
    else
        os << scores[t][WHITE] << " | " << scores[t][BLACK];

    os << " | " << scores[t][WHITE] - scores[t][BLACK] << "\n";
    return os;
  }
}

using namespace Trace;

namespace {

  // Threshold for lazy and space evaluation
  constexpr Value LazyThreshold1 =  Value(1565);
  constexpr Value LazyThreshold2 =  Value(1102);
  constexpr Value SpaceThreshold = Value(11551);
  constexpr Value NNUEThreshold1 =   Value(682);
  constexpr Value NNUEThreshold2 =   Value(176);

  // KingAttackWeights[PieceType] contains king attack weights by piece type
  constexpr int KingAttackWeights[PIECE_TYPE_NB] = { 0, 0, 81, 52, 44, 10 };

  // SafeCheck[PieceType][single/multiple] contains safe check bonus by piece type,
  // higher if multiple safe checks are possible for that piece type.
  constexpr int SafeCheck[][2] = {
      {}, {}, {803, 1292}, {639, 974}, {1087, 1878}, {759, 1132}
  };

#define S(mg, eg) make_score(mg, eg)

  // MobilityBonus[PieceType-2][attacked] contains bonuses for middle and end game,
  // indexed by piece type and number of attacked squares in the mobility area.
  constexpr Score MobilityBonus[][32] = {
    { S(-62,-79), S(-53,-57), S(-12,-31), S( -3,-17), S(  3,  7), S( 12, 13), // Knight
      S( 21, 16), S( 28, 21), S( 37, 26) },
    { S(-47,-59), S(-20,-25), S( 14, -8), S( 29, 12), S( 39, 21), S( 53, 40), // Bishop
      S( 53, 56), S( 60, 58), S( 62, 65), S( 69, 72), S( 78, 78), S( 83, 87),
      S( 91, 88), S( 96, 98) },
    { S(-60,-82), S(-24,-15), S(  0, 17) ,S(  3, 43), S(  4, 72), S( 14,100), // Rook
      S( 20,102), S( 30,122), S( 41,133), S(41 ,139), S( 41,153), S( 45,160),
      S( 57,165), S( 58,170), S( 67,175) },
    { S(-29,-49), S(-16,-29), S( -8, -8), S( -8, 17), S( 18, 39), S( 25, 54), // Queen
      S( 23, 59), S( 37, 73), S( 41, 76), S( 54, 95), S( 65, 95) ,S( 68,101),
      S( 69,124), S( 70,128), S( 70,132), S( 70,133) ,S( 71,136), S( 72,140),
      S( 74,147), S( 76,149), S( 90,153), S(104,169), S(105,171), S(106,171),
      S(112,178), S(114,185), S(114,187), S(119,221) }
  };

  // BishopPawns[distance from edge] contains a file-dependent penalty for pawns on
  // squares of the same color as our bishop.
  constexpr Score BishopPawns[int(FILE_NB) / 2] = {
    S(3, 8), S(3, 9), S(1, 8), S(3, 7)
  };

  // KingProtector[knight/bishop] contains penalty for each distance unit to own king
  constexpr Score KingProtector[] = { S(8, 9), S(6, 9) };

  // Outpost[knight/bishop] contains bonuses for each knight or bishop occupying a
  // pawn protected square on rank 4 to 6 which is also safe from a pawn attack.
  constexpr Score Outpost[] = { S(56, 34), S(31, 23) };

  // PassedRank[Rank] contains a bonus according to the rank of a passed pawn
  constexpr Score PassedRank[RANK_NB] = {
    S(0, 0), S(9, 28), S(15, 31), S(17, 39), S(64, 70), S(171, 177), S(277, 260)
  };

  // RookOnFile[semiopen/open] contains bonuses for each rook when there is
  // no (friendly) pawn on the rook file.
  constexpr Score RookOnFile[] = { S(19, 7), S(48, 27) };

  // ThreatByMinor/ByRook[attacked PieceType] contains bonuses according to
  // which piece type attacks which one. Attacks on lesser pieces which are
  // pawn-defended are not considered.
  constexpr Score ThreatByMinor[PIECE_TYPE_NB] = {
    S(0, 0), S(5, 32), S(55, 41), S(77, 56), S(89, 119), S(79, 162)
  };

  constexpr Score ThreatByRook[PIECE_TYPE_NB] = {
    S(0, 0), S(3, 44), S(37, 68), S(42, 60), S(0, 39), S(58, 43)
  };

  // Assorted bonuses and penalties
  constexpr Score BadOutpost          = S( -7, 36);
  constexpr Score BishopOnKingRing    = S( 24,  0);
  constexpr Score BishopXRayPawns     = S(  4,  5);
  constexpr Score CorneredBishop      = S( 50, 50);
  constexpr Score FlankAttacks        = S(  8,  0);
  constexpr Score Hanging             = S( 69, 36);
  constexpr Score KnightOnQueen       = S( 16, 11);
  constexpr Score LongDiagonalBishop  = S( 45,  0);
  constexpr Score MinorBehindPawn     = S( 18,  3);
  constexpr Score PassedFile          = S( 11,  8);
  constexpr Score PawnlessFlank       = S( 17, 95);
  constexpr Score ReachableOutpost    = S( 31, 22);
  constexpr Score RestrictedPiece     = S(  7,  7);
  constexpr Score RookOnKingRing      = S( 16,  0);
  constexpr Score SliderOnQueen       = S( 60, 18);
  constexpr Score ThreatByKing        = S( 24, 89);
  constexpr Score ThreatByPawnPush    = S( 48, 39);
  constexpr Score ThreatBySafePawn    = S(173, 94);
  constexpr Score TrappedRook         = S( 55, 13);
  constexpr Score WeakQueenProtection = S( 14,  0);
  constexpr Score WeakQueen           = S( 56, 15);


#undef S

  // Evaluation class computes and stores attacks tables and other working data
  template<Tracing T>
  class Evaluation {

  public:
    Evaluation() = delete;
    explicit Evaluation(const Position& p) : pos(p) {}
    Evaluation& operator=(const Evaluation&) = delete;
    Value value();

  private:
    template<Color Us> void initialize();
    template<Color Us, PieceType Pt> Score pieces();
    template<Color Us> Score king() const;
    template<Color Us> Score threats() const;
    template<Color Us> Score passed() const;
    template<Color Us> Score space() const;
    Value winnable(Score score) const;

    const Position& pos;
    Material::Entry* me;
    Pawns::Entry* pe;
    Bitboard mobilityArea[COLOR_NB];
    Score mobility[COLOR_NB] = { SCORE_ZERO, SCORE_ZERO };

    // attackedBy[color][piece type] is a bitboard representing all squares
    // attacked by a given color and piece type. Special "piece types" which
    // is also calculated is ALL_PIECES.
    Bitboard attackedBy[COLOR_NB][PIECE_TYPE_NB];

    // attackedBy2[color] are the squares attacked by at least 2 units of a given
    // color, including x-rays. But diagonal x-rays through pawns are not computed.
    Bitboard attackedBy2[COLOR_NB];

    // kingRing[color] are the squares adjacent to the king plus some other
    // very near squares, depending on king position.
    Bitboard kingRing[COLOR_NB];

    // kingAttackersCount[color] is the number of pieces of the given color
    // which attack a square in the kingRing of the enemy king.
    int kingAttackersCount[COLOR_NB];

    // kingAttackersWeight[color] is the sum of the "weights" of the pieces of
    // the given color which attack a square in the kingRing of the enemy king.
    // The weights of the individual piece types are given by the elements in
    // the KingAttackWeights array.
    int kingAttackersWeight[COLOR_NB];

    // kingAttacksCount[color] is the number of attacks by the given color to
    // squares directly adjacent to the enemy king. Pieces which attack more
    // than one square are counted multiple times. For instance, if there is
    // a white knight on g5 and black's king is on g8, this white knight adds 2
    // to kingAttacksCount[WHITE].
    int kingAttacksCount[COLOR_NB];
  };


  // Evaluation::initialize() computes king and pawn attacks, and the king ring
  // bitboard for a given color. This is done at the beginning of the evaluation.

  template<Tracing T> template<Color Us>
  void Evaluation<T>::initialize() {

    constexpr Color     Them = ~Us;
    constexpr Direction Up   = pawn_push(Us);
    constexpr Direction Down = -Up;
    constexpr Bitboard LowRanks = (Us == WHITE ? Rank2BB | Rank3BB : Rank7BB | Rank6BB);

    const Square ksq = pos.square<KING>(Us);

    Bitboard dblAttackByPawn = pawn_double_attacks_bb<Us>(pos.pieces(Us, PAWN));

    // Find our pawns that are blocked or on the first two ranks
    Bitboard b = pos.pieces(Us, PAWN) & (shift<Down>(pos.pieces()) | LowRanks);

    // Squares occupied by those pawns, by our king or queen, by blockers to attacks on our king
    // or controlled by enemy pawns are excluded from the mobility area.
    mobilityArea[Us] = ~(b | pos.pieces(Us, KING, QUEEN) | pos.blockers_for_king(Us) | pe->pawn_attacks(Them));

    // Initialize attackedBy[] for king and pawns
    attackedBy[Us][KING] = attacks_bb<KING>(ksq);
    attackedBy[Us][PAWN] = pe->pawn_attacks(Us);
    attackedBy[Us][ALL_PIECES] = attackedBy[Us][KING] | attackedBy[Us][PAWN];
    attackedBy2[Us] = dblAttackByPawn | (attackedBy[Us][KING] & attackedBy[Us][PAWN]);

    // Init our king safety tables
    Square s = make_square(std::clamp(file_of(ksq), FILE_B, FILE_G),
                           std::clamp(rank_of(ksq), RANK_2, RANK_7));
    kingRing[Us] = attacks_bb<KING>(s) | s;

    kingAttackersCount[Them] = popcount(kingRing[Us] & pe->pawn_attacks(Them));
    kingAttacksCount[Them] = kingAttackersWeight[Them] = 0;

    // Remove from kingRing[] the squares defended by two pawns
    kingRing[Us] &= ~dblAttackByPawn;
  }


  // Evaluation::pieces() scores pieces of a given color and type

  template<Tracing T> template<Color Us, PieceType Pt>
  Score Evaluation<T>::pieces() {

    constexpr Color     Them = ~Us;
    constexpr Direction Down = -pawn_push(Us);
    constexpr Bitboard OutpostRanks = (Us == WHITE ? Rank4BB | Rank5BB | Rank6BB
                                                   : Rank5BB | Rank4BB | Rank3BB);
    const Square* pl = pos.squares<Pt>(Us);

    Bitboard b, bb;
    Score score = SCORE_ZERO;

    attackedBy[Us][Pt] = 0;

    for (Square s = *pl; s != SQ_NONE; s = *++pl)
    {
        // Find attacked squares, including x-ray attacks for bishops and rooks
        b = Pt == BISHOP ? attacks_bb<BISHOP>(s, pos.pieces() ^ pos.pieces(QUEEN))
          : Pt ==   ROOK ? attacks_bb<  ROOK>(s, pos.pieces() ^ pos.pieces(QUEEN) ^ pos.pieces(Us, ROOK))
                         : attacks_bb<Pt>(s, pos.pieces());

        if (pos.blockers_for_king(Us) & s)
            b &= line_bb(pos.square<KING>(Us), s);

        attackedBy2[Us] |= attackedBy[Us][ALL_PIECES] & b;
        attackedBy[Us][Pt] |= b;
        attackedBy[Us][ALL_PIECES] |= b;

        if (b & kingRing[Them])
        {
            kingAttackersCount[Us]++;
            kingAttackersWeight[Us] += KingAttackWeights[Pt];
            kingAttacksCount[Us] += popcount(b & attackedBy[Them][KING]);
        }

        else if (Pt == ROOK && (file_bb(s) & kingRing[Them]))
            score += RookOnKingRing;

        else if (Pt == BISHOP && (attacks_bb<BISHOP>(s, pos.pieces(PAWN)) & kingRing[Them]))
            score += BishopOnKingRing;

        int mob = popcount(b & mobilityArea[Us]);

        mobility[Us] += MobilityBonus[Pt - 2][mob];

        if (Pt == BISHOP || Pt == KNIGHT)
        {
            // Bonus if the piece is on an outpost square or can reach one
            // Reduced bonus for knights (BadOutpost) if few relevant targets
            bb = OutpostRanks & (attackedBy[Us][PAWN] | shift<Down>(pos.pieces(PAWN)))
                              & ~pe->pawn_attacks_span(Them);
            Bitboard targets = pos.pieces(Them) & ~pos.pieces(PAWN);

            if (   Pt == KNIGHT
                && bb & s & ~CenterFiles // on a side outpost
                && !(b & targets)        // no relevant attacks
                && (!more_than_one(targets & (s & QueenSide ? QueenSide : KingSide))))
                score += BadOutpost;
            else if (bb & s)
                score += Outpost[Pt == BISHOP];
            else if (Pt == KNIGHT && bb & b & ~pos.pieces(Us))
                score += ReachableOutpost;

            // Bonus for a knight or bishop shielded by pawn
            if (shift<Down>(pos.pieces(PAWN)) & s)
                score += MinorBehindPawn;

            // Penalty if the piece is far from the king
            score -= KingProtector[Pt == BISHOP] * distance(pos.square<KING>(Us), s);

            if (Pt == BISHOP)
            {
                // Penalty according to the number of our pawns on the same color square as the
                // bishop, bigger when the center files are blocked with pawns and smaller
                // when the bishop is outside the pawn chain.
                Bitboard blocked = pos.pieces(Us, PAWN) & shift<Down>(pos.pieces());

                score -= BishopPawns[edge_distance(file_of(s))] * pos.pawns_on_same_color_squares(Us, s)
                                     * (!(attackedBy[Us][PAWN] & s) + popcount(blocked & CenterFiles));

                // Penalty for all enemy pawns x-rayed
                score -= BishopXRayPawns * popcount(attacks_bb<BISHOP>(s) & pos.pieces(Them, PAWN));

                // Bonus for bishop on a long diagonal which can "see" both center squares
                if (more_than_one(attacks_bb<BISHOP>(s, pos.pieces(PAWN)) & Center))
                    score += LongDiagonalBishop;

                // An important Chess960 pattern: a cornered bishop blocked by a friendly
                // pawn diagonally in front of it is a very serious problem, especially
                // when that pawn is also blocked.
                if (   pos.is_chess960()
                    && (s == relative_square(Us, SQ_A1) || s == relative_square(Us, SQ_H1)))
                {
                    Direction d = pawn_push(Us) + (file_of(s) == FILE_A ? EAST : WEST);
                    if (pos.piece_on(s + d) == make_piece(Us, PAWN))
                        score -= !pos.empty(s + d + pawn_push(Us))                ? CorneredBishop * 4
                                : pos.piece_on(s + d + d) == make_piece(Us, PAWN) ? CorneredBishop * 2
                                                                                  : CorneredBishop;
                }
            }
        }

        if (Pt == ROOK)
        {
            // Bonus for rook on an open or semi-open file
            if (pos.is_on_semiopen_file(Us, s))
                score += RookOnFile[pos.is_on_semiopen_file(Them, s)];

            // Penalty when trapped by the king, even more if the king cannot castle
            else if (mob <= 3)
            {
                File kf = file_of(pos.square<KING>(Us));
                if ((kf < FILE_E) == (file_of(s) < kf))
                    score -= TrappedRook * (1 + !pos.castling_rights(Us));
            }
        }

        if (Pt == QUEEN)
        {
            // Penalty if any relative pin or discovered attack against the queen
            Bitboard queenPinners;
            if (pos.slider_blockers(pos.pieces(Them, ROOK, BISHOP), s, queenPinners))
                score -= WeakQueen;
        }
    }
    if (T)
        Trace::add(Pt, Us, score);

    return score;
  }


  // Evaluation::king() assigns bonuses and penalties to a king of a given color

  template<Tracing T> template<Color Us>
  Score Evaluation<T>::king() const {

    constexpr Color    Them = ~Us;
    constexpr Bitboard Camp = (Us == WHITE ? AllSquares ^ Rank6BB ^ Rank7BB ^ Rank8BB
                                           : AllSquares ^ Rank1BB ^ Rank2BB ^ Rank3BB);

    Bitboard weak, b1, b2, b3, safe, unsafeChecks = 0;
    Bitboard rookChecks, queenChecks, bishopChecks, knightChecks;
    int kingDanger = 0;
    const Square ksq = pos.square<KING>(Us);

    // Init the score with king shelter and enemy pawns storm
    Score score = pe->king_safety<Us>(pos);

    // Attacked squares defended at most once by our queen or king
    weak =  attackedBy[Them][ALL_PIECES]
          & ~attackedBy2[Us]
          & (~attackedBy[Us][ALL_PIECES] | attackedBy[Us][KING] | attackedBy[Us][QUEEN]);

    // Analyse the safe enemy's checks which are possible on next move
    safe  = ~pos.pieces(Them);
    safe &= ~attackedBy[Us][ALL_PIECES] | (weak & attackedBy2[Them]);

    b1 = attacks_bb<ROOK  >(ksq, pos.pieces() ^ pos.pieces(Us, QUEEN));
    b2 = attacks_bb<BISHOP>(ksq, pos.pieces() ^ pos.pieces(Us, QUEEN));

    // Enemy rooks checks
    rookChecks = b1 & attackedBy[Them][ROOK] & safe;
    if (rookChecks)
        kingDanger += SafeCheck[ROOK][more_than_one(rookChecks)];
    else
        unsafeChecks |= b1 & attackedBy[Them][ROOK];

    // Enemy queen safe checks: count them only if the checks are from squares from
    // which opponent cannot give a rook check, because rook checks are more valuable.
    queenChecks =  (b1 | b2) & attackedBy[Them][QUEEN] & safe
                 & ~(attackedBy[Us][QUEEN] | rookChecks);
    if (queenChecks)
        kingDanger += SafeCheck[QUEEN][more_than_one(queenChecks)];

    // Enemy bishops checks: count them only if they are from squares from which
    // opponent cannot give a queen check, because queen checks are more valuable.
    bishopChecks =  b2 & attackedBy[Them][BISHOP] & safe
                  & ~queenChecks;
    if (bishopChecks)
        kingDanger += SafeCheck[BISHOP][more_than_one(bishopChecks)];

    else
        unsafeChecks |= b2 & attackedBy[Them][BISHOP];

    // Enemy knights checks
    knightChecks = attacks_bb<KNIGHT>(ksq) & attackedBy[Them][KNIGHT];
    if (knightChecks & safe)
        kingDanger += SafeCheck[KNIGHT][more_than_one(knightChecks & safe)];
    else
        unsafeChecks |= knightChecks;

    // Find the squares that opponent attacks in our king flank, the squares
    // which they attack twice in that flank, and the squares that we defend.
    b1 = attackedBy[Them][ALL_PIECES] & KingFlank[file_of(ksq)] & Camp;
    b2 = b1 & attackedBy2[Them];
    b3 = attackedBy[Us][ALL_PIECES] & KingFlank[file_of(ksq)] & Camp;

    int kingFlankAttack  = popcount(b1) + popcount(b2);
    int kingFlankDefense = popcount(b3);

    kingDanger +=        kingAttackersCount[Them] * kingAttackersWeight[Them] // (~10 Elo)
                 + 185 * popcount(kingRing[Us] & weak)                        // (~15 Elo)
                 + 148 * popcount(unsafeChecks)                               // (~4 Elo)
                 +  98 * popcount(pos.blockers_for_king(Us))                  // (~2 Elo)
                 +  69 * kingAttacksCount[Them]                               // (~0.5 Elo)
                 +   3 * kingFlankAttack * kingFlankAttack / 8                // (~0.5 Elo)
                 +       mg_value(mobility[Them] - mobility[Us])              // (~0.5 Elo)
                 - 873 * !pos.count<QUEEN>(Them)                              // (~24 Elo)
                 - 100 * bool(attackedBy[Us][KNIGHT] & attackedBy[Us][KING])  // (~5 Elo)
                 -   6 * mg_value(score) / 8                                  // (~8 Elo)
                 -   4 * kingFlankDefense                                     // (~5 Elo)
                 +  37;                                                       // (~0.5 Elo)

    // Transform the kingDanger units into a Score, and subtract it from the evaluation
    if (kingDanger > 100)
        score -= make_score(kingDanger * kingDanger / 4096, kingDanger / 16);

    // Penalty when our king is on a pawnless flank
    if (!(pos.pieces(PAWN) & KingFlank[file_of(ksq)]))
        score -= PawnlessFlank;

    // Penalty if king flank is under attack, potentially moving toward the king
    score -= FlankAttacks * kingFlankAttack;

    if (T)
        Trace::add(KING, Us, score);

    return score;
  }


  // Evaluation::threats() assigns bonuses according to the types of the
  // attacking and the attacked pieces.

  template<Tracing T> template<Color Us>
  Score Evaluation<T>::threats() const {

    constexpr Color     Them     = ~Us;
    constexpr Direction Up       = pawn_push(Us);
    constexpr Bitboard  TRank3BB = (Us == WHITE ? Rank3BB : Rank6BB);

    Bitboard b, weak, defended, nonPawnEnemies, stronglyProtected, safe;
    Score score = SCORE_ZERO;

    // Non-pawn enemies
    nonPawnEnemies = pos.pieces(Them) & ~pos.pieces(PAWN);

    // Squares strongly protected by the enemy, either because they defend the
    // square with a pawn, or because they defend the square twice and we don't.
    stronglyProtected =  attackedBy[Them][PAWN]
                       | (attackedBy2[Them] & ~attackedBy2[Us]);

    // Non-pawn enemies, strongly protected
    defended = nonPawnEnemies & stronglyProtected;

    // Enemies not strongly protected and under our attack
    weak = pos.pieces(Them) & ~stronglyProtected & attackedBy[Us][ALL_PIECES];

    // Bonus according to the kind of attacking pieces
    if (defended | weak)
    {
        b = (defended | weak) & (attackedBy[Us][KNIGHT] | attackedBy[Us][BISHOP]);
        while (b)
            score += ThreatByMinor[type_of(pos.piece_on(pop_lsb(&b)))];

        b = weak & attackedBy[Us][ROOK];
        while (b)
            score += ThreatByRook[type_of(pos.piece_on(pop_lsb(&b)))];

        if (weak & attackedBy[Us][KING])
            score += ThreatByKing;

        b =  ~attackedBy[Them][ALL_PIECES]
           | (nonPawnEnemies & attackedBy2[Us]);
        score += Hanging * popcount(weak & b);

        // Additional bonus if weak piece is only protected by a queen
        score += WeakQueenProtection * popcount(weak & attackedBy[Them][QUEEN]);
    }

    // Bonus for restricting their piece moves
    b =   attackedBy[Them][ALL_PIECES]
       & ~stronglyProtected
       &  attackedBy[Us][ALL_PIECES];
    score += RestrictedPiece * popcount(b);

    // Protected or unattacked squares
    safe = ~attackedBy[Them][ALL_PIECES] | attackedBy[Us][ALL_PIECES];

    // Bonus for attacking enemy pieces with our relatively safe pawns
    b = pos.pieces(Us, PAWN) & safe;
    b = pawn_attacks_bb<Us>(b) & nonPawnEnemies;
    score += ThreatBySafePawn * popcount(b);

    // Find squares where our pawns can push on the next move
    b  = shift<Up>(pos.pieces(Us, PAWN)) & ~pos.pieces();
    b |= shift<Up>(b & TRank3BB) & ~pos.pieces();

    // Keep only the squares which are relatively safe
    b &= ~attackedBy[Them][PAWN] & safe;

    // Bonus for safe pawn threats on the next move
    b = pawn_attacks_bb<Us>(b) & nonPawnEnemies;
    score += ThreatByPawnPush * popcount(b);

    // Bonus for threats on the next moves against enemy queen
    if (pos.count<QUEEN>(Them) == 1)
    {
        bool queenImbalance = pos.count<QUEEN>() == 1;

        Square s = pos.square<QUEEN>(Them);
        safe =   mobilityArea[Us]
              & ~pos.pieces(Us, PAWN)
              & ~stronglyProtected;

        b = attackedBy[Us][KNIGHT] & attacks_bb<KNIGHT>(s);

        score += KnightOnQueen * popcount(b & safe) * (1 + queenImbalance);

        b =  (attackedBy[Us][BISHOP] & attacks_bb<BISHOP>(s, pos.pieces()))
           | (attackedBy[Us][ROOK  ] & attacks_bb<ROOK  >(s, pos.pieces()));

        score += SliderOnQueen * popcount(b & safe & attackedBy2[Us]) * (1 + queenImbalance);
    }

    if (T)
        Trace::add(THREAT, Us, score);

    return score;
  }

  // Evaluation::passed() evaluates the passed pawns and candidate passed
  // pawns of the given color.

  template<Tracing T> template<Color Us>
  Score Evaluation<T>::passed() const {

    constexpr Color     Them = ~Us;
    constexpr Direction Up   = pawn_push(Us);
    constexpr Direction Down = -Up;

    auto king_proximity = [&](Color c, Square s) {
      return std::min(distance(pos.square<KING>(c), s), 5);
    };

    Bitboard b, bb, squaresToQueen, unsafeSquares, blockedPassers, helpers;
    Score score = SCORE_ZERO;

    b = pe->passed_pawns(Us);

    blockedPassers = b & shift<Down>(pos.pieces(Them, PAWN));
    if (blockedPassers)
    {
        helpers =  shift<Up>(pos.pieces(Us, PAWN))
                 & ~pos.pieces(Them)
                 & (~attackedBy2[Them] | attackedBy[Us][ALL_PIECES]);

        // Remove blocked candidate passers that don't have help to pass
        b &=  ~blockedPassers
            | shift<WEST>(helpers)
            | shift<EAST>(helpers);
    }

    while (b)
    {
        Square s = pop_lsb(&b);

        assert(!(pos.pieces(Them, PAWN) & forward_file_bb(Us, s + Up)));

        int r = relative_rank(Us, s);

        Score bonus = PassedRank[r];

        if (r > RANK_3)
        {
            int w = 5 * r - 13;
            Square blockSq = s + Up;

            // Adjust bonus based on the king's proximity
            bonus += make_score(0, (  king_proximity(Them, blockSq) * 19 / 4
                                    - king_proximity(Us,   blockSq) *  2) * w);

            // If blockSq is not the queening square then consider also a second push
            if (r != RANK_7)
                bonus -= make_score(0, king_proximity(Us, blockSq + Up) * w);

            // If the pawn is free to advance, then increase the bonus
            if (pos.empty(blockSq))
            {
                squaresToQueen = forward_file_bb(Us, s);
                unsafeSquares = passed_pawn_span(Us, s);

                bb = forward_file_bb(Them, s) & pos.pieces(ROOK, QUEEN);

                if (!(pos.pieces(Them) & bb))
                    unsafeSquares &= attackedBy[Them][ALL_PIECES];

                // If there are no enemy attacks on passed pawn span, assign a big bonus.
                // Otherwise assign a smaller bonus if the path to queen is not attacked
                // and even smaller bonus if it is attacked but block square is not.
                int k = !unsafeSquares                    ? 35 :
                        !(unsafeSquares & squaresToQueen) ? 20 :
                        !(unsafeSquares & blockSq)        ?  9 :
                                                             0 ;

                // Assign a larger bonus if the block square is defended
                if ((pos.pieces(Us) & bb) || (attackedBy[Us][ALL_PIECES] & blockSq))
                    k += 5;

                bonus += make_score(k * w, k * w);
            }
        } // r > RANK_3

        score += bonus - PassedFile * edge_distance(file_of(s));
    }

    if (T)
        Trace::add(PASSED, Us, score);

    return score;
  }


  // Evaluation::space() computes a space evaluation for a given side, aiming to improve game
  // play in the opening. It is based on the number of safe squares on the four central files
  // on ranks 2 to 4. Completely safe squares behind a friendly pawn are counted twice.
  // Finally, the space bonus is multiplied by a weight which decreases according to occupancy.

  template<Tracing T> template<Color Us>
  Score Evaluation<T>::space() const {

    // Early exit if, for example, both queens or 6 minor pieces have been exchanged
    if (pos.non_pawn_material() < SpaceThreshold)
        return SCORE_ZERO;

    constexpr Color Them     = ~Us;
    constexpr Direction Down = -pawn_push(Us);
    constexpr Bitboard SpaceMask =
      Us == WHITE ? CenterFiles & (Rank2BB | Rank3BB | Rank4BB)
                  : CenterFiles & (Rank7BB | Rank6BB | Rank5BB);

    // Find the available squares for our pieces inside the area defined by SpaceMask
    Bitboard safe =   SpaceMask
                   & ~pos.pieces(Us, PAWN)
                   & ~attackedBy[Them][PAWN];

    // Find all squares which are at most three squares behind some friendly pawn
    Bitboard behind = pos.pieces(Us, PAWN);
    behind |= shift<Down>(behind);
    behind |= shift<Down+Down>(behind);

    int bonus = popcount(safe) + popcount(behind & safe & ~attackedBy[Them][ALL_PIECES]);
    int weight = pos.count<ALL_PIECES>(Us) - 3 + std::min(pe->blocked_count(), 9);
    Score score = make_score(bonus * weight * weight / 16, 0);

    if (T)
        Trace::add(SPACE, Us, score);

    return score;
  }


  // Evaluation::winnable() adjusts the midgame and endgame score components, based on
  // the known attacking/defending status of the players. The final value is derived
  // by interpolation from the midgame and endgame values.

  template<Tracing T>
  Value Evaluation<T>::winnable(Score score) const {

    int outflanking =  distance<File>(pos.square<KING>(WHITE), pos.square<KING>(BLACK))
                     - distance<Rank>(pos.square<KING>(WHITE), pos.square<KING>(BLACK));

    bool pawnsOnBothFlanks =   (pos.pieces(PAWN) & QueenSide)
                            && (pos.pieces(PAWN) & KingSide);

    bool almostUnwinnable =   outflanking < 0
                           && !pawnsOnBothFlanks;

    bool infiltration =   rank_of(pos.square<KING>(WHITE)) > RANK_4
                       || rank_of(pos.square<KING>(BLACK)) < RANK_5;

    // Compute the initiative bonus for the attacking side
    int complexity =   9 * pe->passed_count()
                    + 12 * pos.count<PAWN>()
                    +  9 * outflanking
                    + 21 * pawnsOnBothFlanks
                    + 24 * infiltration
                    + 51 * !pos.non_pawn_material()
                    - 43 * almostUnwinnable
                    -110 ;

    Value mg = mg_value(score);
    Value eg = eg_value(score);

    // Now apply the bonus: note that we find the attacking side by extracting the
    // sign of the midgame or endgame values, and that we carefully cap the bonus
    // so that the midgame and endgame scores do not change sign after the bonus.
    int u = ((mg > 0) - (mg < 0)) * std::clamp(complexity + 50, -abs(mg), 0);
    int v = ((eg > 0) - (eg < 0)) * std::max(complexity, -abs(eg));

    mg += u;
    eg += v;

    // Compute the scale factor for the winning side
    Color strongSide = eg > VALUE_DRAW ? WHITE : BLACK;
    int sf = me->scale_factor(pos, strongSide);

    // If scale factor is not already specific, scale down via general heuristics
    if (sf == SCALE_FACTOR_NORMAL)
    {
        if (pos.opposite_bishops())
        {
            if (   pos.non_pawn_material(WHITE) == BishopValueMg
                && pos.non_pawn_material(BLACK) == BishopValueMg)
                sf = 18 + 4 * popcount(pe->passed_pawns(strongSide));
            else
                sf = 22 + 3 * pos.count<ALL_PIECES>(strongSide);
        }
        else if (  pos.non_pawn_material(WHITE) == RookValueMg
                && pos.non_pawn_material(BLACK) == RookValueMg
                && pos.count<PAWN>(strongSide) - pos.count<PAWN>(~strongSide) <= 1
                && bool(KingSide & pos.pieces(strongSide, PAWN)) != bool(QueenSide & pos.pieces(strongSide, PAWN))
                && (attacks_bb<KING>(pos.square<KING>(~strongSide)) & pos.pieces(~strongSide, PAWN)))
            sf = 36;
        else if (pos.count<QUEEN>() == 1)
            sf = 37 + 3 * (pos.count<QUEEN>(WHITE) == 1 ? pos.count<BISHOP>(BLACK) + pos.count<KNIGHT>(BLACK)
                                                        : pos.count<BISHOP>(WHITE) + pos.count<KNIGHT>(WHITE));
        else
            sf = std::min(sf, 36 + 7 * pos.count<PAWN>(strongSide)) - 4 * !pawnsOnBothFlanks;

        sf -= 4 * !pawnsOnBothFlanks;
    }

    // Interpolate between the middlegame and (scaled by 'sf') endgame score
    v =  mg * int(me->game_phase())
       + eg * int(PHASE_MIDGAME - me->game_phase()) * ScaleFactor(sf) / SCALE_FACTOR_NORMAL;
    v /= PHASE_MIDGAME;

    if (T)
    {
        Trace::add(WINNABLE, make_score(u, eg * ScaleFactor(sf) / SCALE_FACTOR_NORMAL - eg_value(score)));
        Trace::add(TOTAL, make_score(mg, eg * ScaleFactor(sf) / SCALE_FACTOR_NORMAL));
    }

    return Value(v);
  }


  // Evaluation::value() is the main function of the class. It computes the various
  // parts of the evaluation and returns the value of the position from the point
  // of view of the side to move.

  template<Tracing T>
  Value Evaluation<T>::value() {

    assert(!pos.checkers());

    // Probe the material hash table
    me = Material::probe(pos);

    // If we have a specialized evaluation function for the current material
    // configuration, call it and return.
    if (me->specialized_eval_exists())
        return me->evaluate(pos);

    // Initialize score by reading the incrementally updated scores included in
    // the position object (material + piece square tables) and the material
    // imbalance. Score is computed internally from the white point of view.
    Score score = pos.psq_score() + me->imbalance() + pos.this_thread()->contempt;

    // Probe the pawn hash table
    pe = Pawns::probe(pos);
    score += pe->pawn_score(WHITE) - pe->pawn_score(BLACK);

    // Early exit if score is high
    auto lazy_skip = [&](Value lazyThreshold) {
        return abs(mg_value(score) + eg_value(score)) / 2 > lazyThreshold + pos.non_pawn_material() / 64;
    };

    if (lazy_skip(LazyThreshold1))
        goto make_v;

    // Main evaluation begins here
    initialize<WHITE>();
    initialize<BLACK>();

    // Pieces evaluated first (also populates attackedBy, attackedBy2).
    // Note that the order of evaluation of the terms is left unspecified.
    score +=  pieces<WHITE, KNIGHT>() - pieces<BLACK, KNIGHT>()
            + pieces<WHITE, BISHOP>() - pieces<BLACK, BISHOP>()
            + pieces<WHITE, ROOK  >() - pieces<BLACK, ROOK  >()
            + pieces<WHITE, QUEEN >() - pieces<BLACK, QUEEN >();

    score += mobility[WHITE] - mobility[BLACK];

    // More complex interactions that require fully populated attack bitboards
    score +=  king<   WHITE>() - king<   BLACK>()
            + passed< WHITE>() - passed< BLACK>();

    if (lazy_skip(LazyThreshold2))
        goto make_v;

    score +=  threats<WHITE>() - threats<BLACK>()
            + space<  WHITE>() - space<  BLACK>();

make_v:
    // Derive single value from mg and eg parts of score
    Value v = winnable(score);

    // In case of tracing add all remaining individual evaluation terms
    if (T)
    {
        Trace::add(MATERIAL, pos.psq_score());
        Trace::add(IMBALANCE, me->imbalance());
        Trace::add(PAWN, pe->pawn_score(WHITE), pe->pawn_score(BLACK));
        Trace::add(MOBILITY, mobility[WHITE], mobility[BLACK]);
    }

    // Evaluation grain
    v = (v / 16) * 16;

    // Side to move point of view
    v = (pos.side_to_move() == WHITE ? v : -v) + Tempo;

    return v;
  }

} // namespace


/// evaluate() is the evaluator for the outer world. It returns a static
/// evaluation of the position from the point of view of the side to move.

Value Eval::evaluate(const Position& pos) {

  Value v;

  if (NNUE::useNNUE == NNUE::UseNNUEMode::Pure) {
      v = NNUE::evaluate(pos);

      // Guarantee evaluation does not hit the tablebase range
      v = std::clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);

      return v;
  }
  else if (NNUE::useNNUE == NNUE::UseNNUEMode::False)
      v = Evaluation<NO_TRACE>(pos).value();
  else
  {
      // Scale and shift NNUE for compatibility with search and classical evaluation
      auto  adjusted_NNUE = [&](){
         int mat = pos.non_pawn_material() + PawnValueMg * pos.count<PAWN>();
         return NNUE::evaluate(pos) * (720 + mat / 32) / 1024 + Tempo;
      };

      // If there is PSQ imbalance use classical eval, with small probability if it is small
      Value psq = Value(abs(eg_value(pos.psq_score())));
      int   r50 = 16 + pos.rule50_count();
      bool  largePsq = psq * 16 > (NNUEThreshold1 + pos.non_pawn_material() / 64) * r50;
      bool  classical = largePsq || (psq > PawnValueMg / 4 && !(pos.this_thread()->nodes & 0xB));

      bool strongClassical = pos.non_pawn_material() < 2 * RookValueMg && pos.count<PAWN>() < 2;

      v = classical || strongClassical ? Evaluation<NO_TRACE>(pos).value() : adjusted_NNUE();

      // If the classical eval is small and imbalance large, use NNUE nevertheless.
      // For the case of opposite colored bishops, switch to NNUE eval with
      // small probability if the classical eval is less than the threshold.
      if (   largePsq && !strongClassical
          && (   abs(v) * 16 < NNUEThreshold2 * r50
              || (   pos.opposite_bishops()
                  && abs(v) * 16 < (NNUEThreshold1 + pos.non_pawn_material() / 64) * r50
                  && !(pos.this_thread()->nodes & 0xB))))
          v = adjusted_NNUE();
  }

  // Damp down the evaluation linearly when shuffling
  v = v * (100 - pos.rule50_count()) / 100;

  // Guarantee evaluation does not hit the tablebase range
  v = std::clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);

  return v;
}

/// evaluate_hybrid() is the evaluator for the outer world. It returns a static
/// evaluation of the position from the point of view of the side to move.

Value Eval::evaluate_hybrid(const Position& pos) {

  Value v;

  // Scale and shift NNUE for compatibility with search and classical evaluation
  auto  adjusted_NNUE = [&](){
     int mat = pos.non_pawn_material() + PawnValueMg * pos.count<PAWN>();
     return NNUE::evaluate(pos) * (720 + mat / 32) / 1024 + Tempo;
  };

  // If there is PSQ imbalance use classical eval, with small probability if it is small
  Value psq = Value(abs(eg_value(pos.psq_score())));
  int   r50 = 16 + pos.rule50_count();
  bool  largePsq = psq * 16 > (NNUEThreshold1 + pos.non_pawn_material() / 64) * r50;
  bool  classical = largePsq || (psq > PawnValueMg / 4 && !(pos.this_thread()->nodes & 0xB));

  bool strongClassical = pos.non_pawn_material() < 2 * RookValueMg && pos.count<PAWN>() < 2;

  v = classical || strongClassical ? Evaluation<NO_TRACE>(pos).value() : adjusted_NNUE();

  // If the classical eval is small and imbalance large, use NNUE nevertheless.
  // For the case of opposite colored bishops, switch to NNUE eval with
  // small probability if the classical eval is less than the threshold.
  if (   largePsq && !strongClassical
      && (   abs(v) * 16 < NNUEThreshold2 * r50
          || (   pos.opposite_bishops()
              && abs(v) * 16 < (NNUEThreshold1 + pos.non_pawn_material() / 64) * r50
              && !(pos.this_thread()->nodes & 0xB))))
      v = adjusted_NNUE();

  // Damp down the evaluation linearly when shuffling
  v = v * (100 - pos.rule50_count()) / 100;

  // Guarantee evaluation does not hit the tablebase range
  v = std::clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);

  return v;
}

/// trace() is like evaluate(), but instead of returning a value, it returns
/// a string (suitable for outputting to stdout) that contains the detailed
/// descriptions and values of each evaluation term. Useful for debugging.
/// Trace scores are from white's point of view

std::string Eval::trace(const Position& pos) {

  if (pos.checkers())
      return "Final evaluation: none (in check)";

  std::stringstream ss;
  ss << std::showpoint << std::noshowpos << std::fixed << std::setprecision(2);

  Value v;

  std::memset(scores, 0, sizeof(scores));

  pos.this_thread()->contempt = SCORE_ZERO; // Reset any dynamic contempt

  v = Evaluation<TRACE>(pos).value();

  ss << std::showpoint << std::noshowpos << std::fixed << std::setprecision(2)
     << "     Term    |    White    |    Black    |    Total   \n"
     << "             |   MG    EG  |   MG    EG  |   MG    EG \n"
     << " ------------+-------------+-------------+------------\n"
     << "    Material | " << Term(MATERIAL)
     << "   Imbalance | " << Term(IMBALANCE)
     << "       Pawns | " << Term(PAWN)
     << "     Knights | " << Term(KNIGHT)
     << "     Bishops | " << Term(BISHOP)
     << "       Rooks | " << Term(ROOK)
     << "      Queens | " << Term(QUEEN)
     << "    Mobility | " << Term(MOBILITY)
     << " King safety | " << Term(KING)
     << "     Threats | " << Term(THREAT)
     << "      Passed | " << Term(PASSED)
     << "       Space | " << Term(SPACE)
     << "    Winnable | " << Term(WINNABLE)
     << " ------------+-------------+-------------+------------\n"
     << "       Total | " << Term(TOTAL);

  v = pos.side_to_move() == WHITE ? v : -v;

  ss << "\nClassical evaluation: " << to_cp(v) << " (white side)\n";

  if (NNUE::useNNUE != NNUE::UseNNUEMode::False)
  {
      v = NNUE::evaluate(pos);
      v = pos.side_to_move() == WHITE ? v : -v;
      ss << "\nNNUE evaluation:      " << to_cp(v) << " (white side)\n";
  }

  v = evaluate(pos);
  v = pos.side_to_move() == WHITE ? v : -v;
  ss << "\nFinal evaluation:     " << to_cp(v) << " (white side)\n";

  return ss.str();
}
