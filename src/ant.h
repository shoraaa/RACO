#pragma once

#include "utils.h"


struct RouteIterator {
    const std::vector<uint32_t> &route_;
    size_t position_ = 0;

    uint32_t goto_succ() noexcept {
        position_ = (position_ + 1 < route_.size()) ? position_ + 1 : 0;
        return route_[position_];
    }

    uint32_t goto_pred() noexcept {
        position_ = position_ != 0 ? position_ - 1 : route_.size() - 1;
        return route_[position_];
    }
};


struct Solution {
    std::vector<uint32_t> route_;
    double cost_ = std::numeric_limits<double>::max();
    std::vector<uint32_t> node_indices_;

    Solution() = default;

    Solution(const std::vector<uint32_t> &route, double cost)
        : route_(route),
          cost_(cost),
          node_indices_(route.size(), 0) {
        update_node_indices();
    }

    void update(const std::vector<uint32_t> &route, double cost) {
        route_ = route;
        cost_ = cost;
        update_node_indices();
    }

    void update(const Solution *other) {
        route_ = other->route_;
        cost_ = other->cost_;
        update_node_indices();
    }

    void update_node_indices() {
        for (size_t i = 0; i < route_.size(); ++i) {
            node_indices_[route_[i]] = static_cast<uint32_t>(i);
        }
    }

    // We assume that route is undirected
    [[nodiscard]] bool contains_edge(uint32_t edge_head, uint32_t edge_tail) const {
        return get_succ(edge_head) == edge_tail   // same edge
            || get_pred(edge_head) == edge_tail;  // reversed
    }

    [[nodiscard]] uint32_t get_succ(uint32_t node) const {
        auto index = node_indices_[node];
        return route_[(index + 1u < route_.size()) ? index + 1u : 0u];
    }

    [[nodiscard]] uint32_t get_pred(uint32_t node) const {
        auto index = node_indices_[node];
        return route_[(index > 0u) ? index - 1u : route_.size() - 1u];
    }

    RouteIterator get_iterator(uint32_t start_node) {
        return { route_, node_indices_[start_node] };
    }

};


struct Ant : public Solution {
    using CostFunction = std::function<double (uint32_t, uint32_t)>;

    std::vector<uint32_t> unvisited_;
    Bitmask  visited_bitmask_;
    uint32_t dimension_ = 0;
    uint32_t visited_count_ = 0;
    CostFunction cost_fn_;

    Ant() : Solution() {}

    Ant(const std::vector<uint32_t> &route, double cost)
        : Solution(route, cost),
          unvisited_(route.size(), static_cast<uint32_t >(route.size())),
          dimension_(static_cast<uint32_t>(route.size())),
          visited_count_(static_cast<uint32_t>(route.size())) {
    }

    void initialize(uint32_t dimension) {
        dimension_ = dimension;
        visited_count_ = 0;

        route_.resize(dimension);

        unvisited_.resize(dimension);
        std::iota(unvisited_.begin(), unvisited_.end(), 0);

        visited_bitmask_.resize(dimension);
        visited_bitmask_.clear();
    }

    void visit(uint32_t node) {
        assert(!is_visited(node));

        route_[visited_count_++] = node;
        visited_bitmask_.set_bit(node);
    }

    [[nodiscard]] bool is_visited(uint32_t node) const {
        return visited_bitmask_.get_bit(node);
    }

    bool try_visit(uint32_t node) {
        if (!is_visited(node)) {
            visit(node);
            return true;
        }
        return false;
    }

    [[nodiscard]] uint32_t get_current_node() const {
        return route_[visited_count_ - 1];
    }

    [[nodiscard]] uint32_t get_unvisited_count() const {
        return dimension_ - visited_count_;
    }

    const std::vector<uint32_t> &get_unvisited_nodes() {
        // Filter out visited nodes from unvisited_ list that
        // now can be invalid
        //
        // This has linear complexity but should not be a problem
        // if this method is not called very often.
        size_t n = unvisited_.size();
        size_t j = 0;
        for (size_t i = 0; i < n; ++i) {
            auto node = unvisited_[i];
            if (!is_visited(node)) {
                unvisited_[j++] = node;
            }
        }
        assert(j == get_unvisited_count());
        unvisited_.resize(j);
        return unvisited_;
    }

    void relocate_node(uint32_t target, uint32_t node) {
        assert(node != target);
        assert(node < route_.size());
        assert(target < route_.size());

        if (get_succ(target) == node) { return ; }

        const auto node_pos = node_indices_[node];
        const auto target_pos = node_indices_[target];
        const auto len = route_.size();

        const auto node_pred = get_pred(node);
        const auto node_succ = get_succ(node);
        const auto target_succ = get_succ(target);

        if (target_pos < node_pos) {  // Case 1.
            // 1 2 3 t 5 6 n 7 8 =>
            // 1 2 3 t n 5 6 7 8
            auto beg = route_.rbegin() + len - 1 - node_pos;
            auto end = route_.rbegin() + len - 1 - target_pos;

            std::rotate(beg, beg + 1, end);

            for (auto i = target_pos; i <= node_pos; ++i) {
                node_indices_[route_[i]] = i;
            }
        } else { // Case 2.
            // 1 2 3 n 5 6 t 7 8 =>
            // 1 2 3 5 6 t n 7 8
            auto beg = route_.begin() + node_pos;
            auto end = route_.begin() + target_pos + 1;
            std::rotate(beg, beg + 1, end);

            for (auto i = node_pos; i <= target_pos; ++i) {
                node_indices_[route_[i]] = i;
            }
        }

        assert(get_succ(target) == node);

        // We are removing these edges:
        cost_ += - cost_fn_(node_pred, node)
                 - cost_fn_(node, node_succ)
                 - cost_fn_(target, target_succ)
                 + cost_fn_(node_pred, node_succ)
                 + cost_fn_(target, node)
                 + cost_fn_(node, target_succ);
    }

    size_t size() const { return route_.size(); }
    
    double get_dist_to_succ(uint32_t node) {
        return cost_fn_(node, get_succ(node));
    }

    void two_opt_nn(const ProblemInstance &instance,
                    std::vector<uint32_t> &checklist,
                    uint32_t nn_list_size) {

        // We assume symmetry so that the order of the nodes does not matter
        assert(instance.is_symmetric_);

        // Setting maximum number of allowed route changes prevents very long-running times
        // for very hard to solve TSP instances.
        const uint32_t MaxChanges = size();
        uint32_t changes_count = 0;

        double cost_change = 0;

        size_t checklist_pos_pos = 0;
        while (checklist_pos_pos < checklist.size() && changes_count < MaxChanges) {
            auto a = checklist[checklist_pos_pos++];
            assert(a < route_.size());

            auto a_next = get_succ(a);
            auto a_prev = get_pred(a);

            auto dist_a_to_next = get_dist_to_succ(a);// instance.get_distance(a, a_next);
            auto dist_a_to_prev = get_dist_to_succ(a_prev);

            double max_diff = -1;
            std::array<uint32_t, 4> move;

            const auto &nn_list = instance.get_nearest_neighbors(a, nn_list_size);

            for (auto b : nn_list) {
                auto dist_ab = instance.get_distance(a, b);
                if (dist_a_to_next > dist_ab) {
                    // We rotate the section between a and b_next so that
                    // two new (undirected) edges are created: { a, b } and { a_next, b_next }
                    //
                    // a -> a_next ... b -> b_next
                    // a -> b ... a_next -> b_next
                    //
                    // or
                    //
                    // b -> b_next ... a -> a_next
                    // b -> a ... b_next -> a_next
                    auto b_next = get_succ(b);

                    auto diff = dist_a_to_next
                            + get_dist_to_succ(b) //instance.get_distance(b, b_next)
                            - dist_ab
                            - instance.get_distance(a_next, b_next);

                    if (diff > max_diff) {
                        move = { a_next, b_next, a, b };
                        max_diff = diff;
                    }
                } else {
                    break ;
                }
            }

            for (auto b : nn_list) {
                auto dist_ab = instance.get_distance(a, b);
                if (dist_a_to_prev > dist_ab) {
                    // We rotate the section between a_prev and b so that
                    // two new (undirected) edges are created: { a, b } and { a_prev, b_prev }
                    //
                    // a_prev -> a ... b_prev -> b
                    // a_prev -> b_prev ... a -> b
                    //
                    // or
                    //
                    // b_prev -> b ... a_prev -> a
                    // b_prev -> a_prev ... b -> a
                    auto b_prev = get_pred(b);

                    auto diff = dist_a_to_prev
                            + get_dist_to_succ(b_prev)
                            - dist_ab
                            - instance.get_distance(a_prev, b_prev);

                    if (diff > max_diff) {
                        move = { a, b, a_prev, b_prev };
                        max_diff = diff;
                    }
                } else {
                    break ;
                }
            }

            if (max_diff > 0) {
                flip_route_section(move[0], move[1]);

                for (auto x : move) {
                    if (std::find(checklist.begin() + static_cast<int32_t>(checklist_pos_pos),
                                checklist.end(), x) == checklist.end()) {
                        checklist.push_back(x);
                    }
                }
                ++changes_count;
                cost_change -= max_diff;
            }
        }
        assert(instance.is_route_valid(route_));
        cost_ += cost_change;
    }

        int32_t flip_route_section(int32_t start_node, int32_t end_node) {
        auto first = node_indices_[start_node];
        auto last = node_indices_[end_node];

        if (first > last) {
            std::swap(first, last);
        }

        const auto length = static_cast<int32_t>(route_.size());
        const int32_t segment_length = last - first;
        const int32_t remaining_length = length - segment_length;

        if (segment_length <= remaining_length) {  // Reverse the specified segment
            std::reverse(route_.begin() + first, route_.begin() + last);

            for (auto k = first; k < last; ++k) {
                node_indices_[ route_[k] ] = k;
            }
            return first;
        } else {  // Reverse the rest of the route, leave the segment intact
            first = (first > 0) ? first - 1 : length - 1;
            last = last % length;
            std::swap(first, last);
            int32_t l = first;
            int32_t r = last;
            int32_t i = 0;
            int32_t j = length - first + last + 1;
            while(i++ < j--) {
                std::swap(route_[l], route_[r]);
                node_indices_[route_[l]] = l;
                node_indices_[route_[r]] = r;
                l = (l+1) % length;
                r = (r > 0) ? r - 1 : length - 1;
            }
        }
        return 0;
    }


};
