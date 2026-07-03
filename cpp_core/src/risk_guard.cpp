#include "../include/risk_guard.h"
#include <cmath>

RiskGuard::RiskGuard(double max_pos, double max_spread)
    : max_position_units_(max_pos), max_allowable_spread_bps_(max_spread), current_position_units_(0.0) {}

bool RiskGuard::validate_quote(double price, double size, bool is_buy) const {
    // 1. Defend against negative or corrupt pricing inputs
    if (price <= 0.0 || size <= 0.0) return false;

    // 2. Simulated Positional Pre-Trade Check
    double projected_position = current_position_units_ + (is_buy ? size : -size);
    if (std::abs(projected_position) > max_position_units_) {
        return false; // Hard reject: Position limits would be breached
    }

    return true;
}

void RiskGuard::update_position(double size, bool is_buy) {
    current_position_units_ += (is_buy ? size : -size);
}

// Add this implementation to physically enforce the limit
bool RiskGuard::validate_spread(double spread_bps) const {
    return spread_bps <= max_allowable_spread_bps_;
}
