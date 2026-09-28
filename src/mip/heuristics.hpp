// SPDX-License-Identifier: Apache-2.0
// NIRNAY - primal heuristics for branch and bound: the feasibility pump and RINS.
//
// References:
//   Fischetti, Glover & Lodi, "The feasibility pump", Mathematical Programming 104(1), 2005 -
//     alternate between rounding an LP point and projecting the rounding back onto the LP
//     polyhedron in the L1 distance, with random flips to escape cycles.
//   Bertacco, Fischetti & Lodi, "A feasibility pump heuristic for general mixed-integer
//     problems", Discrete Optimization 4(1), 2007 - the distance term for integer columns at
//     a bound, which is the form used here (interior general integers carry no weight).
//   Danna, Rothberg & Le Pape, "Exploring relaxation induced neighborhoods to improve MIP
//     solutions", Mathematical Programming 102(1), 2005 - RINS: fix every integer column on
//     which the incumbent and the node relaxation agree, and search what is left.
//
// WHY. A branch-and-bound proof is only as fast as its incumbent is good: can_prune() fathoms
// against it, and a search with no incumbent fathoms nothing. The root dive finds one on
// models with structure; these two cover what it misses. The pump targets models where no
// rounding of the relaxation is feasible (MIPLIB enlight8: 0 incumbents in 60 s); RINS turns
// an incumbent the tree found into a better one, which is where most of an optimality gap is
// closed in practice.
//
// NOTHING HERE IS TRUSTED. Every candidate goes back through the search's offer_incumbent(),
// which checks integrality, column bounds and every row against the ORIGINAL model before a
// point becomes the incumbent.
#pragma once

#include <functional>
#include <vector>

#include "nirnay/logging.hpp"
#include "nirnay/model.hpp"
#include "nirnay/options.hpp"
#include "nirnay/solve_control.hpp"

namespace nirnay::mip {

/// Hands a candidate to the search; returns true when it became the new incumbent.
using OfferIncumbent = std::function<bool(const std::vector<double>&)>;

/// The feasibility pump from the LP point `start`. Runs at most `max_rounds` projections or
/// `seconds` of wall-clock, and stops at the first candidate `offer` accepts. Returns the
/// number of distance LPs solved.
int feasibility_pump(const Model& model, const std::vector<double>& start, const Options& options,
                     int max_rounds, double seconds, const OfferIncumbent& offer,
                     SolveControl* control);

/// RINS around `incumbent` and the relaxation point `relaxation`: a sub-MIP of `model` with
/// every agreeing integer column fixed and the objective cut off at the incumbent, searched
/// for at most `node_limit` nodes and `seconds`. Declines (returns false without solving)
/// when fewer than `min_fixed_fraction` of the integer columns agree. Returns true when it
/// offered a point that `offer` accepted.
bool rins(const Model& model, const std::vector<double>& incumbent,
          const std::vector<double>& relaxation, const Options& options, Count node_limit,
          double seconds, double min_fixed_fraction, const OfferIncumbent& offer,
          SolveControl* control);

}  // namespace nirnay::mip
