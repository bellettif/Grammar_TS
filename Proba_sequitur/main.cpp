#include <iostream>
#include <string>
#include <regex>
#include <iterator>
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include <random>
#include <cstdlib>      // std::rand, std::srand

#include <boost/algorithm/string.hpp>

#include "string_utils.h"
#include "reduce_utils.h"
#include "divergence_metrics.h"
#include "preprocessing.h"
#include "decision_making.h"
#include "proba_sequitur.h"
#include "data.h";

int main ()
{

    /*
    std::vector<std::string> sequences = {"a a a a b c a a b c",
                                          "a a b b c d a d b c",
                                          "e d d e a a a b c b c",
                                          "d b c a a a b c a a a",
                                          "a a c b c a a b c"};

    std::vector<std::string> sequences_for_counts = {"a a a a b c a a b c",
                                                     "a a b b c d a d b c",
                                                     "e d d e a a a b c b c",
                                                     "d b c a a a b c a a a",
                                                     "a a c b c a a b c",
                                                     "a a e d a a d b c",
                                                     "b b a b e d b",
                                                     "a d d e e d d",
                                                     "a b b c c d"};
    */

    std::vector<std::string> sequences = {achu_1,
                                         achu_2,
                                         achu_3,
                                         achu_4,
                                         achu_5,
                                         achu_6,
                                         achu_7,
                                         achu_8,
                                         achu_9};

    std::vector<std::string> sequences_for_counts = sequences;

    int degree = 9;
    int max_rules = 40;
    bool atomic_bare_lk = true;
    bool stochastic = false;

    Proba_sequitur ps (sequences,
                       sequences_for_counts,
                       degree,
                       max_rules,
                       atomic_bare_lk,
                       stochastic);

    ps.run();

    ps.print_sequences_for_counts();


    return 0;
}
