#pragma once

#include <assert.h>

#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <sstream>

namespace poker_dice {

using Action = int;

// All actions, but the liar call, could be represented as (quantity, face)
// pair.
struct UnpackedAction {
  int quantity, face;
};

#define CALL_ID 4
#define P0_FOLD_ID 2
#define P1_FOLD_ID 3

// Public state of the game without tracking the history of the game.
struct PartialPublicState {
  // Previous call.
  Action last_bid;
  int event = -1; // 0 for fold, 1 for call
  // Player to make move next.
  int player_id;

  int hand;

  bool operator==(const PartialPublicState& state) const {
    return last_bid == state.last_bid && player_id == state.player_id && event == state.event && hand == state.hand;
  }
};


class Game {
 public:
  const int num_dice = 2;
  const int num_faces = 6;
  const int max_bid = 9;
  const int call_action = 1;
  const int raise_action = 2; // raise by 1
  const int fold_action = 0; // game over
  
  int score_table[6*6*6*6*6] ;

  Game(int num_dice, int num_faces)
      : 
        num_hands_(int_pow(num_faces, num_dice))
         {build_score_table();}

  // Number of dice for all the players.
  //int total_num_dice() const { return total_num_dice_; } // TODO

  // Maximum number of distinct actions in every node.
  Action num_actions() const { return 3; }

  // Number of distrinct game states at the beginning of the game. In other
  // words, number of different realization of the chance nodes.
  int num_hands() const { return num_hands_; }

  // Upper bound for how deep game tree could be.
  int max_depth() const { return max_bid; } // TODO

  void build_score_table()
  {
    // single     3 bits High card 1-6
    // pair       6 bits next bit  1-6
    // 3 kind     3 bits next bit  1-6
    // straight   3 bits next bit  1-6  7 if full house
    // full house <7 on previous bit>
    // 4x         3 bits 1-6
    // 5x         3 bits 1-6

    for (int d0=0 ; d0 < num_faces ;++d0){
    for (int d1=0 ; d1 < num_faces ;++d1){
    for (int d2=0 ; d2 < num_faces ;++d2){
    for (int d3=0 ; d3 < num_faces ;++d3){
    for (int d4=0 ; d4 < num_faces ;++d4){
      int score = 0;
      int triple_num{0}, double_high_num{0}, double_low_num{0}, single_num{0}, 
          straight_num{0}, fours_num{0}, fives_num{0}; 
     
      // sort
      int d[5] = {d0, d1, d2, d3, d4};
      std::sort(d, d+5);

      // check for straight
      if ((d[4] == d[3]+1) && (d[3] == d[2]+1) && (d[2] == d[1]+1) && (d[1] == d[0]+1))
      {
        straight_num = d[0] + 1; // lowest number in straight
      }
      else
      {
        // check for fives
        if (d[0] == d[4])
        {
          fives_num = d[0] + 1;
        }
        // check for fours
        else if (d[0] == d[3])
        {
          fours_num = d[0] + 1;
          single_num = d[4] + 1;
        }
        else if (d[1] == d[4])
        {
          fours_num = d[1] + 1;
          single_num = d[0] + 1;
        }
        else
        {
          // rest: two pair / full house / triple / one pair / bust
          // start with triple:
          if (d[0] == d[2])
          {
            triple_num = d[0] + 1;
            if (d[3] == d[4]) {double_high_num = d[3] + 1; straight_num = 7;} // full house
            else single_num = d[4] + 1;
          }
          else if (d[1] == d[3])
          {
            triple_num = d[1] + 1;
            single_num = d[4] + 1;
          }
          else if (d[2] == d[4])
          {
            triple_num = d[2] + 1;
            if (d[0] == d[1]) {double_high_num = d[1] + 1; straight_num = 7;} // full house
            else single_num = d[1] + 1;
          }

          // no triple, no full house
          else
          {
            // how many pairs?
            int num_pairs = 0;
            std::vector<int> pair_i;
            if (d[0] == d[1]) {num_pairs++; pair_i.push_back(0);}
            if (d[1] == d[2]) {num_pairs++; pair_i.push_back(1);}
            if (d[2] == d[3]) {num_pairs++; pair_i.push_back(2);}
            if (d[3] == d[4]) {num_pairs++; pair_i.push_back(3);}

            if (num_pairs == 0) single_num = d[4] + 1; // bust
            else if (num_pairs == 1) // one pair
            {
              double_low_num = d[pair_i[0]] + 1;
              single_num = (pair_i[0] == 3 ? d[2] : d[4]) + 1;
            }
            else
            {
              double_low_num = (d[pair_i[0]] > d[pair_i[1]]) ? d[pair_i[1]] + 1 : d[pair_i[0]]+1;
              double_high_num = (d[pair_i[0]] > d[pair_i[1]]) ? d[pair_i[0]] + 1 : d[pair_i[1]]+1;
              single_num = d[(0+1+2+3+4 - pair_i[0]*2 - 1 - pair_i[1]*2 - 1)] + 1;
            }
          }
        }
      }

      score += fives_num; score = score << 3;
      score += fours_num; score = score << 3;
      score += straight_num; score = score << 3;
      score += triple_num; score = score << 3;
      score += double_high_num; score = score << 3;
      score += double_low_num; score = score << 3;
      score += single_num;

      score_table[d0+d1*6+d2*36+d3*36*6+d4*36*36] = score;
    }}}}}
  }

  void print_score(double s) const
  {
      int tmp; int ss =s;

      std::cerr << "[";
      for (int i = 0 ; i < 7 ; i++)
      {
        tmp = ss & 7;
        std::cerr << tmp << ",";
        ss = ss >> 3;
      }
      std::cerr << "]\n";
  }

  int utility(int * hands)
  {
    // for test only
    return utility(hands[0]*6+hands[1],hands[2]*6+hands[3],hands[4]*36+hands[5]*6+hands[6]);
  }

  int score(int hand, int public_hand) const
  {
    //int public_die_0 = public_hand % num_faces;
    //int public_die_1 = (public_hand / num_faces) % num_faces;
    //int public_die_2 =  (public_hand / num_faces) / num_faces;
    //int private_die_0 = hand % num_faces;
    //int private_die_1 = hand / num_faces;

    // single
    // pair
    // 3 kind
    // straight
    // full house
    // 4x
    // 5x

    return score_table[public_hand * num_faces * num_faces + hand];
  }

  double utility(int myhand, int ophand, int public_hand) const
  {
    double my_score = score(myhand, public_hand);
    double op_score = score(ophand, public_hand);
    //print_score(my_score);
    //print_score(op_score);
    if (op_score == my_score) return 0.5;
    return (op_score > my_score) ? 0 : 1;
  }

  // player 0 initially bids 1
  // player 1 initially bids 2

  PartialPublicState get_initial_state(int hand) const {
    PartialPublicState state;
    state.last_bid = 2;
    state.player_id = 0;
    state.hand = hand;
    return state;
  }

  // Get range of possible actions in the state as [min_action, max_action).
  std::pair<Action, Action> get_bid_range(
      const PartialPublicState& state) const {
    if (state.event > -1) return std::pair<Action,Action>(0,0);
    return state.last_bid == max_bid
               ? std::pair<Action, Action>(0,2)                // las bid was max, call or fold
               : std::pair<Action, Action>(0,3);
  }

  bool is_terminal(const PartialPublicState& state) const {
    return state.event > -1;
  }

  PartialPublicState act(const PartialPublicState& state, Action action) const {
    const auto range = get_bid_range(state);
    assert(action >= range.first);
    assert(action < range.second);
    PartialPublicState new_state;
    new_state.hand = state.hand;
    if (action == fold_action)
    {
      new_state.last_bid = state.last_bid;
      new_state.event = 0;
    }
    else if (action == call_action)
    {
      new_state.last_bid = state.last_bid;
      new_state.event = 1;
    }
    else // raise_action
    {
      new_state.last_bid = state.last_bid+1;
    }
    new_state.player_id = 1 - state.player_id;
    
    return new_state;
  }

  //std::string action_to_string(Action action) const;
  //std::string state_to_string(const PartialPublicState& state) const;
  //std::string action_to_string_short(Action action) const;
  //std::string state_to_string_short(const PartialPublicState& state) const;


  std::string action_to_string(Action action) const {
    if (action == call_action) {
      return "call";
    }
    if (action == raise_action) {
      return "raise";
    }
    if (action == fold_action) {
      return "fold";
    }
  }

  std::string state_to_string(const PartialPublicState& state) const {
    std::ostringstream ss;
    std::string es = "raise"; if (state.event == 0) es = "fold"; else if (state.event == 1) es = "call";
    ss << "(pid=" << state.player_id << ",pub-hand=" << state.hand << ",last=" << state.last_bid << ",event=" << es << ")";
    return ss.str();
  }

  std::string action_to_string_short(Action action) const {
    return action_to_string(action);
  }

  std::string state_to_string_short(const PartialPublicState& state) const {
    return state_to_string(state);
  }

  Action deduce_last_action(const PartialPublicState& current_state, const PartialPublicState& /*last_state*/) const
  {
    if (current_state.event < 0) return raise_action;
    else if (current_state.event < 1) return fold_action;
    else return call_action;
  }


 private:
  static int int_pow(int base, int power) {
    if (power == 0) return 1;
    const int half_power = int_pow(base, power / 2);
    const int reminder = (power % 2 == 0) ? 1 : base;
    const double double_half_power = half_power;
    const double double_result =
        double_half_power * double_half_power * reminder;
    assert(double_result + 1 <
           static_cast<double>(std::numeric_limits<int>::max()));
    return half_power * half_power * reminder;
  }

  const int num_hands_;
};

}  // namespace
