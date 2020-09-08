﻿#ifndef _LEARN_H_
#define _LEARN_H_

#if defined(EVAL_LEARN)

#include <vector>

// ----------------------
// Floating point for learning
// ----------------------

// If this is set to double, the calculation accuracy will be higher, but the weight array entangled memory will be doubled.
// Currently, if this is float, the weight array is 4.5 times the size of the evaluation function file. (About 4.5GB with KPPT)
// Even if it is a double type, there is almost no difference in the way of convergence, so fix it to float.

// when using float
typedef float LearnFloatType;

// when using double
//typedef double LearnFloatType;

// when using float16
//#include "half_float.h"
//typedef HalfFloat::float16 LearnFloatType;

// ======================
// configure
// ======================

// ----------------------
// Learning with the method of elmo (WCSC27)
// ----------------------

#define LOSS_FUNCTION "ELMO_METHOD(WCSC27)"

// ----------------------
// Definition of struct used in Learner
// ----------------------
#include "../position.h"

namespace Learner
{
	// ----------------------
	// Settings for learning
	// ----------------------

	// mini-batch size.
	// Calculate the gradient by combining this number of phases.
	// If you make it smaller, the number of update_weights() will increase and the convergence will be faster. The gradient is incorrect.
	// If you increase it, the number of update_weights() decreases, so the convergence will be slow. The slope will come out accurately.
	// I don't think you need to change this value in most cases.

	constexpr std::size_t LEARN_MINI_BATCH_SIZE = 1000 * 1000 * 1;

	// The number of phases to read from the file at one time. After reading this much, shuffle.
	// It is better to have a certain size, but this number x 40 bytes x 3 times as much memory is consumed. 400MB*3 is consumed in the 10M phase.
	// Must be a multiple of THREAD_BUFFER_SIZE(=10000).

	constexpr std::size_t LEARN_SFEN_READ_SIZE = 1000 * 1000 * 10;

	// Saving interval of evaluation function at learning. Save each time you learn this number of phases.
	// Needless to say, the longer the saving interval, the shorter the learning time.
	// Folder name is incremented for each save like 0/, 1/, 2/...
	// By default, once every 1 billion phases.
	constexpr std::size_t LEARN_EVAL_SAVE_INTERVAL = 1000000000ULL;

	// Reduce the output of rmse during learning to 1 for this number of times.
	// rmse calculation is done in one thread, so it takes some time, so reducing the output is effective.
	constexpr std::size_t LEARN_RMSE_OUTPUT_INTERVAL = 1;

	//Structure in which PackedSfen and evaluation value are integrated
	// If you write different contents for each option, it will be a problem when reusing the teacher game
	// For the time being, write all the following members regardless of the options.
	struct PackedSfenValue
	{
		// phase
		PackedSfen sfen;

		// Evaluation value returned from Learner::search()
		int16_t score;

		// PV first move
		// Used when finding the match rate with the teacher
		uint16_t move;

		// Trouble of the phase from the initial phase.
		uint16_t gamePly;

		// 1 if the player on this side ultimately wins the game. -1 if you are losing.
		// 0 if a draw is reached.
		// The draw is in the teacher position generation command gensfen,
		// Only write if LEARN_GENSFEN_DRAW_RESULT is enabled.
		int8_t game_result;

		// When exchanging the file that wrote the teacher aspect with other people
		//Because this structure size is not fixed, pad it so that it is 40 bytes in any environment.
		uint8_t padding;

		// 32 + 2 + 2 + 2 + 1 + 1 = 40bytes
	};

	// Type that returns the reading line and the evaluation value at that time
	// Used in Learner::search(), Learner::qsearch().
	typedef std::pair<Value, std::vector<Move> > ValueAndPV;

	// Phase array: PSVector stands for packed sfen vector.
	typedef std::vector<PackedSfenValue> PSVector;

	// So far, only Yaneura King 2018 Otafuku has this stub
	// This stub is required if EVAL_LEARN is defined.
	extern Learner::ValueAndPV  search(Position& pos, int depth , size_t multiPV = 1 , uint64_t NodesLimit = 0);
	extern Learner::ValueAndPV qsearch(Position& pos);

	double calc_grad(Value shallow, const PackedSfenValue& psv);
	
	void convert_bin_from_pgn_extract(
		const std::vector<std::string>& filenames,
		const std::string& output_file_name,
		const bool pgn_eval_side_to_move,
		const bool convert_no_eval_fens_as_score_zero);
	
	void convert_bin(
		const std::vector<std::string>& filenames,
		const std::string& output_file_name,
		const int ply_minimum,
		const int ply_maximum,
		const int interpolate_eval,
		const int src_score_min_value,
		const int src_score_max_value,
		const int dest_score_min_value,
		const int dest_score_max_value,
		const bool check_invalid_fen,
		const bool check_illegal_move);

	void convert_plain(
		const std::vector<std::string>& filenames,
		const std::string& output_file_name);
}

#endif

#endif // ifndef _LEARN_H_
