#ifndef NNUE_BUCKET_H_INCLUDED
#define NNUE_BUCKET_H_INCLUDED

#include "nnue/nnue_common.h"

#include "position.h"

namespace Eval::NNUE {

    inline IndexType get_nnue_bucket_from_counts(
        int wq, int wr, int wb, int wn, int wp,
        int bq, int br, int bb, int bn, int bp)
    {
        const int phase =
            4 * (wq + bq)
          + 2 * (wr + br)
          +     (wb + bb)
          +     (wn + bn);

        const int white_material =
            9 * wq + 5 * wr + 3 * (wb + wn) + wp;

        const int black_material =
            9 * bq + 5 * br + 3 * (bb + bn) + bp;

        const int material_imbalance =
            white_material >= black_material
                ? white_material - black_material
                : black_material - white_material;

        if (phase <= 10)
            return 3; // reduced-material / endgame-like

        if (material_imbalance >= 3)
            return 2; // materially imbalanced non-endgame

        if (phase >= 18)
            return 0; // balanced high-material non-endgame

        return 1;     // balanced reduced-material non-endgame
    }

    inline IndexType get_nnue_bucket(const Position& pos)
    {
        return get_nnue_bucket_from_counts(
            pos.count<QUEEN>(WHITE),
            pos.count<ROOK>(WHITE),
            pos.count<BISHOP>(WHITE),
            pos.count<KNIGHT>(WHITE),
            pos.count<PAWN>(WHITE),

            pos.count<QUEEN>(BLACK),
            pos.count<ROOK>(BLACK),
            pos.count<BISHOP>(BLACK),
            pos.count<KNIGHT>(BLACK),
            pos.count<PAWN>(BLACK));
    }

} // namespace Eval::NNUE

#endif // NNUE_BUCKET_H_INCLUDED
