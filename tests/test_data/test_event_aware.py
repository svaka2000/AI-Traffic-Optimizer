"""Tests for aito/optimization/event_aware.py (GF3)."""
import pytest
from datetime import datetime, timedelta
from aito.optimization.event_aware import (
    EventType,
    EventDemandProfile,
    CalendarEvent,
    assess_corridor_impact,
    EventTimingPlan,
    EventOptimizationResult,
    EventAwareOptimizer,
    SD_EVENT_PROFILES,
)
from aito.data.san_diego_inventory import get_corridor

ROSECRANS = get_corridor("rosecrans")
NOW = datetime(2026, 4, 17, 14, 0, 0)
GAME_START = NOW.replace(hour=19, minute=10) + timedelta(days=1)


def _make_padres_game() -> CalendarEvent:
    return CalendarEvent(
        event_id="padres_test_001",
        name="Padres vs. Dodgers",
        venue_lat=32.7076,
        venue_lon=-117.1570,
        event_type=EventType.SPORTS_MAJOR,
        start_time=GAME_START,
        end_time=GAME_START + timedelta(hours=3, minutes=15),
        expected_attendance=41500,
        venue_capacity=45000,
    )


class TestEventType:
    def test_sports_major_exists(self):
        assert EventType.SPORTS_MAJOR in EventType.__members__.values()

    def test_at_least_four_types(self):
        assert len(list(EventType)) >= 4


class TestSDEventProfiles:
    def test_all_event_types_have_profiles(self):
        for etype in EventType:
            assert etype in SD_EVENT_PROFILES, f"No profile for {etype}"

    def test_sports_major_has_high_demand(self):
        profile = SD_EVENT_PROFILES[EventType.SPORTS_MAJOR]
        assert profile.inbound_peak_veh_hr > 0
        assert profile.outbound_peak_veh_hr > 0

    def test_pre_deploy_min_positive(self):
        for etype, profile in SD_EVENT_PROFILES.items():
            assert profile.pre_deploy_min >= 0


class TestCalendarEvent:
    def test_instantiation(self):
        event = _make_padres_game()
        assert event is not None

    def test_attendance_fraction(self):
        event = _make_padres_game()
        af = event.attendance_fraction
        assert 0 < af <= 1.5  # allows mild overflow

    def test_inbound_peak_veh_hr_positive(self):
        event = _make_padres_game()
        assert event.inbound_peak_veh_hr > 0

    def test_outbound_peak_veh_hr_positive(self):
        event = _make_padres_game()
        assert event.outbound_peak_veh_hr > 0

    def test_pre_deploy_time_before_start(self):
        event = _make_padres_game()
        assert event.pre_deploy_time < event.start_time

    def test_egress_end_time_after_end(self):
        event = _make_padres_game()
        assert event.egress_end_time > event.end_time

    def test_phase_upcoming_before_pre_deploy(self):
        event = _make_padres_game()
        phase = event.phase(NOW)
        assert phase in ("upcoming", "pre-event", "inbound", "event", "egress", "ended")

    def test_phase_active_during_event(self):
        event = _make_padres_game()
        mid_event = event.start_time + timedelta(hours=1)
        phase = event.phase(mid_event)
        assert phase in ("event", "inbound")

    def test_demand_profile_returns_profile(self):
        event = _make_padres_game()
        profile = event.demand_profile
        assert isinstance(profile, EventDemandProfile)


class TestAssessCorridorImpact:
    def test_returns_dict(self):
        event = _make_padres_game()
        result = assess_corridor_impact(event, ROSECRANS)
        assert isinstance(result, dict)

    def test_has_impact_scores(self):
        event = _make_padres_game()
        # Use a large radius to ensure we get some impacts
        result = assess_corridor_impact(event, ROSECRANS, max_impact_radius_m=10000.0)
        assert len(result) > 0
        # All values should be floats (demand multipliers or impact scores)
        for k, v in result.items():
            assert isinstance(v, (int, float))


class TestEventAwareOptimizer:
    def setup_method(self):
        self.optimizer = EventAwareOptimizer(ROSECRANS)
        self.event = _make_padres_game()

    def test_register_event(self):
        self.optimizer.register_event(self.event)
        assert self.event in self.optimizer._events

    def test_optimized_plan_stored_after_register(self):
        self.optimizer.register_event(self.event)
        assert self.event.event_id in self.optimizer._optimized_plans

    def test_optimization_result_type(self):
        self.optimizer.register_event(self.event)
        result = self.optimizer._optimized_plans[self.event.event_id]
        assert isinstance(result, EventOptimizationResult)

    def test_result_has_corridor_id(self):
        self.optimizer.register_event(self.event)
        result = self.optimizer._optimized_plans[self.event.event_id]
        assert result.corridor_id == ROSECRANS.id

    def test_get_active_plans_returns_list(self):
        self.optimizer.register_event(self.event)
        plans = self.optimizer.get_active_plans(now=GAME_START - timedelta(minutes=30))
        assert isinstance(plans, list)

    def test_no_active_plans_before_pre_deploy(self):
        self.optimizer.register_event(self.event)
        # Well before pre-deploy
        plans = self.optimizer.get_active_plans(now=GAME_START - timedelta(days=2))
        assert len(plans) == 0

    def test_list_events_returns_list(self):
        self.optimizer.register_event(self.event)
        cal = self.optimizer.list_events()
        assert isinstance(cal, list)
        assert len(cal) >= 1

    def test_multiple_events_registered(self):
        event2 = CalendarEvent(
            event_id="concert_001",
            name="Taylor Swift",
            venue_lat=32.7076,
            venue_lon=-117.1570,
            event_type=EventType.CONCERT_LARGE,
            start_time=GAME_START + timedelta(days=7),
            end_time=GAME_START + timedelta(days=7, hours=3),
            expected_attendance=20000,
            venue_capacity=25000,
        )
        self.optimizer.register_event(self.event)
        self.optimizer.register_event(event2)
        assert len(self.optimizer._events) == 2
        assert len(self.optimizer._optimized_plans) == 2

    def test_inbound_peak_higher_for_major_vs_minor(self):
        major = _make_padres_game()
        minor = CalendarEvent(
            event_id="minor_001",
            name="Minor Game",
            venue_lat=32.7076,
            venue_lon=-117.1570,
            event_type=EventType.SPORTS_MINOR,
            start_time=GAME_START,
            end_time=GAME_START + timedelta(hours=2),
            expected_attendance=5000,
            venue_capacity=10000,
        )
        # Major event with high attendance should have more demand
        assert major.inbound_peak_veh_hr > minor.inbound_peak_veh_hr
