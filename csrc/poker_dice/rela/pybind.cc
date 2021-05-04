// Copyright (c) Facebook, Inc. and its affiliates.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/extension.h>

#include "real_net.h"
#include "recursive_solving.h"
#include "stats.h"

#include "rela/context.h"
#include "rela/data_loop.h"
#include "rela/prioritized_replay.h"
#include "rela/thread_loop.h"

namespace py = pybind11;
using namespace rela;

namespace {

std::shared_ptr<ThreadLoop> create_cfr_thread(
    std::shared_ptr<ModelLocker> modelLocker,
    std::shared_ptr<ValuePrioritizedReplay> replayBuffer,
    const poker_dice::RecursiveSolvingParams& cfg, int seed) {
  auto connector =
      std::make_shared<CVNetBufferConnector>(modelLocker, replayBuffer);
  return std::make_shared<DataThreadLoop>(std::move(connector), cfg, seed);
}

float compute_exploitability(poker_dice::RecursiveSolvingParams params,
                             const std::string& model_path) {
  py::gil_scoped_release release;
  poker_dice::Game game(params.num_dice, params.num_faces);
  std::shared_ptr<IValueNet> net =
      poker_dice::create_torchscript_net(model_path);
  const auto tree_strategy =
      compute_strategy_recursive(game, params.subgame_params, net);
  poker_dice::print_strategy(game, unroll_tree(game, 152), tree_strategy);
  std::cerr << "DANGEROUS TODO !!!" << std::endl;
  return poker_dice::compute_exploitability(game, tree_strategy, 152);
}

auto compute_stats_with_net(poker_dice::RecursiveSolvingParams params,
                            const std::string& model_path) {
  py::gil_scoped_release release;
  poker_dice::Game game(params.num_dice, params.num_faces);
  std::shared_ptr<IValueNet> net =
      poker_dice::create_torchscript_net(model_path);
  
  //poker_dice::print_strategy(game, unroll_tree(game), net_strategy);

  float explotability_sum = 0;
  for (int pub_hand=0 ; pub_hand < 216 ; pub_hand++)
  {
    const auto net_strategy =
      compute_strategy_recursive_to_leaf(game, params.subgame_params, pub_hand, net);
    float explotability =
      poker_dice::compute_exploitability(game, net_strategy, pub_hand);
    explotability_sum += explotability;
  }


  //return std::make_tuple(explotability_sum / 216, mse_net_traverse, mse_full_traverse);
  return std::make_tuple(explotability_sum / 216, 0., 0.);
}

float compute_full_game_cfr(int pub_hand, int iterations)
{
    poker_dice::RecursiveSolvingParams cfg; cfg.num_dice = 1; cfg.num_faces = 6;
    cfg.subgame_params.max_depth = 100;
    cfg.subgame_params.use_cfr = true;
    cfg.subgame_params.linear_update = true;

    auto runner = std::make_unique<poker_dice::RlRunner>(cfg, nullptr, 3253);

    return runner->step_test(pub_hand, iterations);
}

int get_node_id(const poker_dice::Tree & tree, const poker_dice::PartialPublicState & search_state)
{
  for (size_t node_id = 0; node_id < tree.size(); ++node_id) {
    if (tree[node_id].state == search_state) return node_id;
  }
  return -1;
}

poker_dice::Action get_action_from_strategy(const poker_dice::TreeStrategy & strat, 
    const poker_dice::Tree & tree,
    const poker_dice::Game & game,
    const poker_dice::PartialPublicState & state,
    int hand)
{
  int node_id = get_node_id(tree, state);
  auto action_probs = strat[node_id][hand];
  //int first = game.get_bid_range(state).first;
  int last = game.get_bid_range(state).second;

  double action_weights[3];

  for (int i = 0; i < 3 ; i++)
  {
    if (i < last) action_weights[i] = action_probs[i];
    else action_weights[i] = 0.;
  }

  std::default_random_engine generator;
  std::discrete_distribution<int> distribution {action_weights[0],action_weights[1],action_weights[2]};


  std::cout << "Action Probabilities (Fold, Call, Raise): " << distribution.probabilities() << std::endl;

  poker_dice::Action action = distribution(generator);
  return action;
}

void print_public_hand(int hand)
{
  int i = hand % 6 + 1;
  int j = hand / 6 % 6 + 1;
  int k = hand / 36 + 1;
  std::cout << i << " " << j << " " << k;
}

void print_private_hand(int hand)
{
  int i = hand % 6 + 1;
  int j = hand / 6 + 1;
  std::cout << i << " " << j ;
}

void arbitrate(int* hands, int rand_pub_hand, 
    const poker_dice::Game & game,
    const poker_dice::PartialPublicState & state)
{
  int winner = 0;
  int v = 0;
  if (state.event == 0) //fold
  {
    winner = state.player_id;

    std::cout << "Winner is Player " << winner;
 
    std::cout << " Bet was " << state.last_bid - 1 << std::endl;
  }
  else // call
  {
    v = game.utility(hands[0], hands[1], rand_pub_hand); // return 1 if player 0 has better hand
    if (v == 0.0)
    {
      winner = 1;
      std::cout << "Winner is Player " << winner;
 
      std::cout << " Bet was " << state.last_bid << std::endl;
    }
    else if (v == 1.0)
    {
      winner = 0;
      std::cout << "Winner is Player " << winner;
 
      std::cout << " Bet was " << state.last_bid << std::endl;
    }
    else //tie
    {
      winner = -1;
      std::cout << "Tie\n";
    }
  }
  
}

auto play_poker_dice(poker_dice::RecursiveSolvingParams params,
                            const std::string& model_path)
{
  py::gil_scoped_release release;
  poker_dice::Game game(params.num_dice, params.num_faces);
  std::shared_ptr<IValueNet> net =
      poker_dice::create_torchscript_net(model_path);

  poker_dice::RecursiveSolvingParams cfg; cfg.num_dice = 1; cfg.num_faces = 6;
  cfg.subgame_params.max_depth = 100;
  cfg.subgame_params.use_cfr = true;
  cfg.subgame_params.linear_update = true;
  auto runner = std::make_unique<poker_dice::RlRunner>(cfg, nullptr, 3253);
  
  poker_dice::TreeStrategy ts[2];
  
  while(1)
  {
    std::cin.get();
    std::cout << "\n\n\n\n\nNEW GAME\nNeural Net is Player 0 and Full-game CFR is Player 1\n\n";
    int rand_pub_hand = rand() % 216;
    int hands[2];
    hands[0] = rand() % 36;
    hands[1] = rand() % 36;
    auto state = game.get_initial_state(rand_pub_hand);
    const auto tree = unroll_tree(game, rand_pub_hand);

    //std::cout << "Calculating strategies for hand: " << rand_pub_hand << std::endl;
    ts[0] = compute_strategy_recursive_to_leaf(game, params.subgame_params, rand_pub_hand, net);
    ts[1] = runner->get_full_game_cfr_strategy(rand_pub_hand);

    std::cout << "public hand: ";
    print_public_hand(rand_pub_hand);
    std::cout << std::endl;
    std::cout << "private hand Player 0: ";
    print_private_hand(hands[0]);
    std::cout << std::endl;
    std::cout << "private hand Player 1: ";
    print_private_hand(hands[1]);
    std::cout << std::endl;

    while(!game.is_terminal(state))
    {
      std::cout << "State: " << game.state_to_string(state) << std::endl;

      poker_dice::TreeStrategy strat = ts[state.player_id];
      poker_dice::Action action = get_action_from_strategy(strat, tree, game, state, hands[state.player_id]);

      std::cout << "Player: " << state.player_id << " Action: " << game.action_to_string(action) << std::endl;
      state = game.act(state, action);
      std::cin.get();
    }
    
    arbitrate(hands, rand_pub_hand, game, state);
  }
}

float compute_exploitability_no_net(poker_dice::RecursiveSolvingParams params) {
  py::gil_scoped_release release;
  poker_dice::Game game(params.num_dice, params.num_faces);
  auto fp = poker_dice::build_solver(game, game.get_initial_state(152),
                                     poker_dice::get_initial_beliefs(game),
                                     params.subgame_params, /*net=*/nullptr);
  float values[2] = {0.0};
  for (int iter = 0; iter < params.subgame_params.num_iters; ++iter) {
    if (((iter + 1) & iter) == 0 ||
        iter + 1 == params.subgame_params.num_iters) {
      auto values = compute_exploitability2(game, fp->get_strategy(), 152);
      printf("Iter=%8d exploitabilities=(%.3e, %.3e) sum=%.3e\n", iter + 1,
             values[0], values[1], (values[0] + values[1]) / 2.);
    }
    // Check for Ctrl-C.
    if (PyErr_CheckSignals() != 0) throw py::error_already_set();
  }
  std::cerr << "DANGEROUS TODO !!!" << std::endl;
  poker_dice::print_strategy(game, unroll_tree(game,152), fp->get_strategy());
  return values[0] + values[1];
}

// std::shared_ptr<MyAgent> create_value_policy_agent(
//     std::shared_ptr<ModelLocker> modelLocker,
//     std::shared_ptr<ValuePrioritizedReplay> replayBuffer,
//     std::shared_ptr<ValuePrioritizedReplay> policyReplayBuffer,
//     bool compress_policy_values) {
//   return std::make_shared<MyAgent>(modelLocker, replayBuffer,
//                                    policyReplayBuffer,
//                                    compress_policy_values);
// }

}  // namespace

PYBIND11_MODULE(rela, m) {
  py::class_<ValueTransition, std::shared_ptr<ValueTransition>>(
      m, "ValueTransition")
      .def(py::init<>())
      .def_readwrite("query", &ValueTransition::query)
      .def_readwrite("values", &ValueTransition::values);

  py::class_<ValuePrioritizedReplay, std::shared_ptr<ValuePrioritizedReplay>>(
      m, "ValuePrioritizedReplay")
      .def(py::init<int,    // capacity,
                    int,    // seed,
                    float,  // alpha, priority exponent
                    float,  // beta, importance sampling exponent
                    int, bool, bool>(),
           py::arg("capacity"), py::arg("seed"), py::arg("alpha"),
           py::arg("beta"), py::arg("prefetch"), py::arg("use_priority"),
           py::arg("compressed_values"))
      .def("size", &ValuePrioritizedReplay::size)
      .def("num_add", &ValuePrioritizedReplay::numAdd)
      .def("sample", &ValuePrioritizedReplay::sample)
      .def("pop_until", &ValuePrioritizedReplay::popUntil)
      .def("load", &ValuePrioritizedReplay::load)
      .def("save", &ValuePrioritizedReplay::save)
      .def("extract", &ValuePrioritizedReplay::extract)
      .def("push", &ValuePrioritizedReplay::push,
           py::call_guard<py::gil_scoped_release>())
      .def("update_priority", &ValuePrioritizedReplay::updatePriority);

  py::class_<ThreadLoop, std::shared_ptr<ThreadLoop>>(m, "ThreadLoop");

  py::class_<poker_dice::SubgameSolvingParams>(m, "SubgameSolvingParams")
      .def(py::init<>())
      .def_readwrite("num_iters", &poker_dice::SubgameSolvingParams::num_iters)
      .def_readwrite("max_depth", &poker_dice::SubgameSolvingParams::max_depth)
      .def_readwrite("linear_update",
                     &poker_dice::SubgameSolvingParams::linear_update)
      .def_readwrite("optimistic",
                     &poker_dice::SubgameSolvingParams::optimistic)
      .def_readwrite("use_cfr", &poker_dice::SubgameSolvingParams::use_cfr)
      .def_readwrite("dcfr", &poker_dice::SubgameSolvingParams::dcfr)
      .def_readwrite("dcfr_alpha",
                     &poker_dice::SubgameSolvingParams::dcfr_alpha)
      .def_readwrite("dcfr_beta", &poker_dice::SubgameSolvingParams::dcfr_beta)
      .def_readwrite("dcfr_gamma",
                     &poker_dice::SubgameSolvingParams::dcfr_gamma);

  py::class_<poker_dice::RecursiveSolvingParams>(m, "RecursiveSolvingParams")
      .def(py::init<>())
      .def_readwrite("num_dice", &poker_dice::RecursiveSolvingParams::num_dice)
      .def_readwrite("num_faces",
                     &poker_dice::RecursiveSolvingParams::num_faces)
      .def_readwrite("random_action_prob",
                     &poker_dice::RecursiveSolvingParams::random_action_prob)
      .def_readwrite("sample_leaf",
                     &poker_dice::RecursiveSolvingParams::sample_leaf)
      .def_readwrite("subgame_params",
                     &poker_dice::RecursiveSolvingParams::subgame_params);

  py::class_<DataThreadLoop, ThreadLoop, std::shared_ptr<DataThreadLoop>>(
      m, "DataThreadLoop")
      .def(py::init<std::shared_ptr<CVNetBufferConnector>,
                    const poker_dice::RecursiveSolvingParams&, int>(),
           py::arg("connector"), py::arg("params"), py::arg("thread_id"));

  py::class_<rela::Context>(m, "Context")
      .def(py::init<>())
      .def("push_env_thread", &rela::Context::pushThreadLoop,
           py::keep_alive<1, 2>())
      .def("start", &rela::Context::start)
      .def("pause", &rela::Context::pause)
      .def("resume", &rela::Context::resume)
      .def("terminate", &rela::Context::terminate)
      .def("terminated", &rela::Context::terminated);

  py::class_<ModelLocker, std::shared_ptr<ModelLocker>>(m, "ModelLocker")
      .def(py::init<std::vector<py::object>, const std::string&>())
      .def("update_model", &ModelLocker::updateModel);

  m.def("compute_exploitability_fp", &compute_exploitability_no_net,
        py::arg("params"));

  m.def("compute_exploitability_with_net", &compute_exploitability,
        py::arg("params"), py::arg("model_path"));

  m.def("compute_stats_with_net", &compute_stats_with_net, py::arg("params"),
        py::arg("model_path"));


  m.def("play_poker_dice", &play_poker_dice, py::arg("params"),
        py::arg("model_path"));


  m.def("compute_full_game_cfr", &compute_full_game_cfr, py::arg("pub_hand"), py::arg("iterations"));

  m.def("create_cfr_thread", &create_cfr_thread, py::arg("model_locker"),
        py::arg("replay"), py::arg("cfg"), py::arg("seed"));

  //   m.def("create_value_policy_agent", &create_value_policy_agent,
  //         py::arg("model_locker"), py::arg("replay"),
  //         py::arg("policy_replay"),
  //         py::arg("compress_policy_values"));
}
