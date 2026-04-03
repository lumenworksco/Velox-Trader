"""ARCH-002: Dependency Injection Container.

Lightweight DI system that holds all shared dependencies and manages their
lifecycle.  Components are lazily created on first access and cached for reuse.

Usage::

    container = Container.instance()
    rm = container.risk_manager
    bus = container.event_bus

Thread safety: the singleton is created under a lock (double-checked locking).
Individual component factories are also guarded so each dependency is created
exactly once even under concurrent access.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class Container:
    """Central dependency container for the trading bot.

    Holds references to all major subsystems.  Dependencies are created lazily
    via factory methods and cached for the lifetime of the container.
    """

    _instance: Container | None = None
    _instance_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def instance(cls) -> Container:
        """Return the global container singleton (thread-safe)."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
                    logger.info("Container: singleton created")
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Tear down the singleton.  Intended for tests only."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance._shutdown()
            cls._instance = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._components: dict[str, Any] = {}
        self._factories: dict[str, Any] = {}
        self._resolving: set[str] = set()  # HIGH-022: cycle detection

        # Register default factories
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default factory functions for known components."""
        self.register_factory("config", self._create_config)
        self.register_factory("database", self._create_database)
        self.register_factory("event_bus", self._create_event_bus)
        self.register_factory("broker_client", self._create_broker_client)
        self.register_factory("data_client", self._create_data_client)
        self.register_factory("risk_manager", self._create_risk_manager)
        self.register_factory("oms", self._create_oms)
        self.register_factory("circuit_breaker", self._create_circuit_breaker)

        # V11 modules — lazy-imported so they only load on first access
        self._register_v11_defaults()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_factory(self, name: str, factory) -> None:
        """Register a factory callable for a named component.

        The factory is called with no arguments and should return the
        component instance.  Overwrites any existing factory.
        """
        with self._lock:
            self._factories[name] = factory
            # Clear cached instance so the new factory takes effect
            self._components.pop(name, None)

    def register_instance(self, name: str, instance: Any) -> None:
        """Register a pre-built instance (useful for testing)."""
        with self._lock:
            self._components[name] = instance

    def get(self, name: str) -> Any:
        """Retrieve a component by name, creating it if necessary."""
        # Fast path — no lock
        inst = self._components.get(name)
        if inst is not None:
            return inst

        with self._lock:
            # Double-check under lock
            inst = self._components.get(name)
            if inst is not None:
                return inst

            factory = self._factories.get(name)
            if factory is None:
                raise KeyError(f"Container: no factory registered for '{name}'")

            # HIGH-022: Cycle detection — prevent infinite recursion
            if name in self._resolving:
                raise RuntimeError(
                    f"Container: circular dependency detected while resolving '{name}'. "
                    f"Resolution chain: {self._resolving}"
                )
            self._resolving.add(name)
            try:
                logger.debug("Container: creating '%s'", name)
                inst = factory()
                self._components[name] = inst
                return inst
            finally:
                self._resolving.discard(name)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def config(self):
        """The global config module."""
        return self.get("config")

    @property
    def database(self):
        """Database connection / helper."""
        return self.get("database")

    @property
    def event_bus(self):
        """The global EventBus instance."""
        return self.get("event_bus")

    @property
    def broker_client(self):
        """Broker abstraction client."""
        return self.get("broker_client")

    @property
    def data_client(self):
        """Market data client."""
        return self.get("data_client")

    @property
    def risk_manager(self):
        """RiskManager instance."""
        return self.get("risk_manager")

    @property
    def oms(self):
        """Order Management System (OrderManager)."""
        return self.get("oms")

    @property
    def circuit_breaker(self):
        """TieredCircuitBreaker instance."""
        return self.get("circuit_breaker")

    # ------------------------------------------------------------------
    # Default factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def _create_config():
        import config as _cfg
        return _cfg

    @staticmethod
    def _create_database():
        import database as _db
        _db.init_db()
        return _db

    @staticmethod
    def _create_event_bus():
        from engine.events import get_event_bus
        return get_event_bus()

    @staticmethod
    def _create_broker_client():
        from broker.base import Broker
        import config as _cfg

        if _cfg.PAPER_MODE:
            from broker.paper_broker import PaperBroker
            return PaperBroker()
        # Default: return the base Broker (Alpaca REST wrapper)
        return Broker()

    @staticmethod
    def _create_data_client():
        import data as _data
        return _data

    @staticmethod
    def _create_risk_manager():
        from risk import RiskManager
        return RiskManager()

    @staticmethod
    def _create_oms():
        from oms import OrderManager
        return OrderManager()

    @staticmethod
    def _create_circuit_breaker():
        from risk.circuit_breaker import TieredCircuitBreaker
        return TieredCircuitBreaker()

    # ------------------------------------------------------------------
    # V11 module factories (ARCH-011)
    # ------------------------------------------------------------------

    def _register_v11_defaults(self) -> None:
        """Register V11 module factories with lazy imports."""

        # --- Risk modules ---
        self.register_factory("intraday_controls", lambda: (
            __import__("risk.intraday_controls", fromlist=["IntradayRiskControls"])
            .IntradayRiskControls()
        ))
        self.register_factory("factor_model", lambda: (
            __import__("risk.factor_model", fromlist=["FactorRiskModel"])
            .FactorRiskModel()
        ))
        self.register_factory("stress_test", lambda: (
            __import__("risk.stress_testing", fromlist=["StressTestFramework"])
            .StressTestFramework()
        ))
        self.register_factory("gap_risk", lambda: (
            __import__("risk.gap_risk", fromlist=["GapRiskManager"])
            .GapRiskManager()
        ))
        self.register_factory("dynamic_hedger", lambda: (
            __import__("risk.dynamic_hedging", fromlist=["DynamicHedger"])
            .DynamicHedger()
        ))
        self.register_factory("margin_monitor", lambda: (
            __import__("risk.margin_monitor", fromlist=["MarginMonitor"])
            .MarginMonitor()
        ))
        self.register_factory("corporate_actions", lambda: (
            __import__("risk.corporate_actions", fromlist=["CorporateActionDetector"])
            .CorporateActionDetector()
        ))
        self.register_factory("conformal_stops", lambda: (
            __import__("risk.conformal_stops", fromlist=["ConformalStopEngine"])
            .ConformalStopEngine()
        ))

        # --- Execution modules ---
        self.register_factory("fill_analytics", lambda: (
            __import__("execution.fill_analytics", fromlist=["FillAnalytics"])
            .FillAnalytics()
        ))
        self.register_factory("slippage_model", lambda: (
            __import__("execution.slippage_model", fromlist=["SlippageModel"])
            .SlippageModel()
        ))

        # --- Microstructure ---
        self.register_factory("vpin", lambda: (
            __import__("microstructure.vpin", fromlist=["VPIN"])
            .VPIN()
        ))

        # --- ML modules ---
        self.register_factory("feature_engine", lambda: (
            __import__("ml.features", fromlist=["FeatureEngine"])
            .FeatureEngine()
        ))
        self.register_factory("batch_inference", lambda: (
            __import__("ml.inference", fromlist=["BatchInferenceEngine"])
            .BatchInferenceEngine()
        ))
        self.register_factory("model_registry", lambda: (
            __import__("ml.model_registry", fromlist=["ModelRegistry"])
            .ModelRegistry()
        ))
        self.register_factory("bocpd", lambda: (
            __import__("ml.change_point", fromlist=["BayesianChangePointDetector"])
            .BayesianChangePointDetector()
        ))

        # --- Data modules ---
        self.register_factory("feature_store", lambda: (
            __import__("data.feature_store", fromlist=["FeatureStore"])
            .FeatureStore()
        ))
        self.register_factory("data_quality", lambda: (
            __import__("data.quality", fromlist=["DataQualityFramework"])
            .DataQualityFramework()
        ))

        # --- Monitoring modules ---
        self.register_factory("alert_manager", lambda: (
            __import__("monitoring.alerting", fromlist=["AlertManager"])
            .AlertManager()
        ))
        self.register_factory("latency_tracker", lambda: (
            __import__("monitoring.latency", fromlist=["LatencyTracker"])
            .LatencyTracker()
        ))
        self.register_factory("metrics_pipeline", lambda: (
            __import__("monitoring.metrics", fromlist=["MetricsPipeline"])
            .MetricsPipeline()
        ))
        self.register_factory("v11_reconciler", lambda: (
            __import__("monitoring.reconciliation", fromlist=["PositionReconciler"])
            .PositionReconciler()
        ))
        self.register_factory("watchdog", lambda: (
            __import__("monitoring.watchdog", fromlist=["Watchdog"])
            .Watchdog()
        ))

        # --- Compliance modules ---
        self.register_factory("audit_trail", lambda: (
            __import__("compliance.audit_trail", fromlist=["AuditTrail"])
            .AuditTrail()
        ))
        self.register_factory("pdt_compliance", lambda: (
            __import__("compliance.pdt", fromlist=["PDTCompliance"])
            .PDTCompliance()
        ))
        self.register_factory("surveillance", lambda: (
            __import__("compliance.surveillance", fromlist=["SelfSurveillance"])
            .SelfSurveillance()
        ))

        # --- Ops modules ---
        self.register_factory("drawdown_risk", lambda: (
            __import__("ops.drawdown_risk", fromlist=["DrawdownRiskManager"])
            .DrawdownRiskManager()
        ))
        self.register_factory("disaster_recovery", lambda: (
            __import__("ops.disaster_recovery", fromlist=["DisasterRecovery"])
            .DisasterRecovery()
        ))

        # --- Alpha modules ---
        self.register_factory("enhanced_seasonality", lambda: (
            __import__("alpha.seasonality", fromlist=["EnhancedSeasonality"])
            .EnhancedSeasonality()
        ))

        # --- Engine modules ---
        self.register_factory("shadow_trader", lambda: (
            __import__("engine.shadow_mode", fromlist=["ShadowTrader"])
            .ShadowTrader()
        ))

    # ------------------------------------------------------------------
    # V11 convenience properties
    # ------------------------------------------------------------------

    @property
    def batch_inference(self):
        """BatchInferenceEngine instance."""
        return self.get("batch_inference")

    @property
    def model_registry(self):
        """ModelRegistry instance."""
        return self.get("model_registry")

    @property
    def watchdog(self):
        """Watchdog instance."""
        return self.get("watchdog")

    @property
    def shadow_trader(self):
        """ShadowTrader instance."""
        return self.get("shadow_trader")

    @property
    def dynamic_hedger(self):
        """DynamicHedger instance."""
        return self.get("dynamic_hedger")

    @property
    def margin_monitor(self):
        """MarginMonitor instance."""
        return self.get("margin_monitor")

    @property
    def corporate_actions(self):
        """CorporateActionDetector instance."""
        return self.get("corporate_actions")

    @property
    def conformal_stops(self):
        """ConformalStopEngine instance."""
        return self.get("conformal_stops")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        """Clean up components that need explicit teardown."""
        logger.info("Container: shutting down")
        self._components.clear()

    def __repr__(self) -> str:
        with self._lock:
            created = list(self._components.keys())
            registered = list(self._factories.keys())
        return (
            f"Container(created={created}, registered={registered})"
        )
