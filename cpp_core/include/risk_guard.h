#pragma once

class RiskGuard {
private:
    const double max_position_units_;
    const double max_allowable_spread_bps_;
    double current_position_units_;

public:
    RiskGuard(double max_pos, double max_spread);

    bool validate_spread(double spread_bps) const;

    // Returns true if the proposed quote changes comply with hard corporate limits
    bool validate_quote(double price, double size, bool is_buy) const;

    // Tracking current inventory state
    void update_position(double size, bool is_buy);
    double get_position() const { return current_position_units_; }
};
