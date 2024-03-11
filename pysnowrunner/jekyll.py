import argparse
import hashlib
import json
import logging
import math
import os
import pprint
import shutil
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from typing import Set  # noqa: F401
from typing import Tuple  # noqa: F401
from typing import Any, Dict, Iterator, List, Optional, Sequence, TypedDict

from .loader import StringData, TruckData
from .types import Engine, Truck, TruckTire


def round_up(x: Decimal) -> Decimal:
    if x == 0:
        return x
    e = int(math.log10(x))
    d = 1
    while d * 10**e < x:
        d += 1
    return Decimal(d * 10**e)


def json_decimal_default(x):
    if isinstance(x, Decimal):
        return float(x)
    else:
        raise TypeError("cannot JSONify %r" % (x,))


class _ChartJsDataset(TypedDict, total=False):
    xAxisID: str
    yAxisID: str


class ChartJsDataset(_ChartJsDataset):
    label: str
    data: List[Decimal]


class JekyllOutput(object):
    def __init__(self, directory: str, strings: Dict[str, str]):
        self.directory = directory
        self.strings = strings

    @staticmethod
    def write_json(path: str, x) -> None:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(
                x,
                fp,
                default=json_decimal_default,
                indent=2,
                separators=(",", ": "),
            )
            fp.write("\n")

    @contextmanager
    def write_data(self, name: str) -> Iterator[str]:
        data = os.path.join(self.directory, "_data")
        os.makedirs(data, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=data, prefix="tmp.") as temp:
            new_name = os.path.join(temp, name)
            yield new_name
            old_name = os.path.join(data, name)
            try:
                os.unlink(old_name)
            except IsADirectoryError:
                shutil.rmtree(old_name)
            except FileNotFoundError:
                pass
            os.rename(new_name, old_name)

    def write_datadir(self, subdir: str, xs: Dict[str, Any]) -> None:
        with self.write_data(subdir) as new_subdir:
            os.makedirs(new_subdir)
            for name, x in xs.items():
                self.write_json(os.path.join(new_subdir, name + ".json"), x)

    def get_display_name(self, uiname: str, fallback: str = "UNNAMED") -> str:
        # missing UI_TIRE_FEMM_37AT_NAME
        return self.strings.get(uiname, "") or uiname or fallback

    @staticmethod
    def engine_chart(
        labels: List[str],
        torques: List[Decimal],
        fuel_consumptions: List[Decimal],
        max_torque: Optional[Decimal] = None,
        max_fuel_consumption: Optional[Decimal] = None,
    ) -> Any:
        return {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [
                    ChartJsDataset(
                        label="Torque",
                        data=torques,
                        xAxisID="x",
                    ),
                    ChartJsDataset(
                        label="Fuel Consumption",
                        data=fuel_consumptions,
                        xAxisID="x2",
                    ),
                ],
            },
            "options": {
                "maintainAspectRatio": False,
                "indexAxis": "y",
                "scales": {
                    "y": {
                        "ticks": {
                            "autoSkip": False,
                            # "maxRotation": 60,
                        },
                        "stepSize": 1,
                    },
                    "x": {
                        "type": "linear",
                        "beginAtZero": True,
                        "max": max_torque,
                        "display": True,
                        "position": "top",
                    },
                    "x2": {
                        "type": "linear",
                        "beginAtZero": True,
                        "max": max_fuel_consumption,
                        "display": True,
                        "position": "bottom",
                        "grid": {
                            "drawOnChartArea": False,
                        },
                    },
                },
            },
        }

    def to_data_engine(
        self,
        slug: str,
        engine: Engine,
        truck_engines: Dict[str, Dict[str, Truck]],
    ) -> Any:
        return {
            "slug": slug,
            "display_name": self.strings[engine.UiName],
            "torque": engine.Torque,
            "fuel_consumption": engine.FuelConsumption,
            "trucks": [
                {
                    "slug": truck_slug,
                    "display_name": self.strings[truck.UiName],
                }
                for truck_slug, truck in truck_engines.get(slug, {}).items()
            ],
            "python": pprint.pformat(engine),
        }

    def engines(
        self,
        engines: Dict[str, Dict[str, Engine]],
        trucks: Dict[str, Truck],
    ) -> None:
        max_torque = Decimal(0)
        max_fuel_consumption = Decimal(0)

        all_engines = []  # type: List[Tuple[str, Engine]]
        for slug, es in engines.items():
            for e in es.values():
                all_engines.append((slug, e))
                max_torque = max(max_torque, e.Torque)
                max_fuel_consumption = max(max_fuel_consumption, e.FuelConsumption)

        truck_engines = {}  # type: Dict[str, Dict[str, Truck]]
        for truck_slug, truck in trucks.items():
            for socket in truck.EngineSocket:
                for engine_slug in socket.Type:
                    truck_engines.setdefault(engine_slug, {})[truck_slug] = truck

        all_engines = sorted(all_engines, key=lambda e: e[1].Torque)
        with self.write_data("engines.json") as engines_file:
            self.write_json(
                engines_file,
                {
                    "chartjs": self.engine_chart(
                        labels=[self.strings[e.UiName] for _, e in all_engines],
                        torques=[e.Torque for _, e in all_engines],
                        fuel_consumptions=[e.FuelConsumption for _, e in all_engines],
                        max_torque=round_up(max_torque),
                        max_fuel_consumption=round_up(max_fuel_consumption),
                    ),
                    "engines": [
                        self.to_data_engine(slug, engine, truck_engines)
                        for slug, engine in all_engines
                    ],
                },
            )

    @staticmethod
    def tire_chart(
        labels: List[str],
        frictions: Tuple[ChartJsDataset, ...],
        min_friction: Optional[Decimal] = None,
        max_friction: Optional[Decimal] = None,
    ) -> Any:
        return {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": frictions,
            },
            "options": {
                "maintainAspectRatio": False,
                "indexAxis": "y",
                "scales": {
                    "y": {
                        "ticks": {
                            "autoSkip": False,
                            # "maxRotation": 60,
                        },
                    },
                    "x": {
                        "min": min_friction,
                        "max": max_friction,
                    },
                },
            },
        }

    def to_data_tire(
        self,
        slug: str,
        tire: TruckTire,
        truck_tires: Dict[str, Dict[str, Truck]],
    ) -> Any:
        return {
            "slug": slug,
            "display_name": self.get_display_name(tire.UiName),
            "asphalt": tire.WheelFriction.BodyFrictionAsphalt,
            "dirt": tire.WheelFriction.BodyFriction,
            "mud": tire.WheelFriction.SubstanceFriction,
            "trucks": [
                {
                    "slug": truck_slug,
                    "display_name": self.strings[truck.UiName],
                }
                for truck_slug, truck in truck_tires.get(slug, {}).items()
            ],
            "python": pprint.pformat(tire),
        }

    def tires(
        self,
        tires: Dict[str, Dict[str, TruckTire]],
        trucks: Dict[str, Truck],
    ) -> None:
        all_tires = []  # type: List[Tuple[str, TruckTire]]
        for slug, ts in tires.items():
            for t in ts.values():
                all_tires.append((slug, t))

        truck_tires = {}  # type: Dict[str, Dict[str, Truck]]
        for truck_slug, truck in trucks.items():
            for w in truck.CompatibleWheels:
                truck_tires.setdefault(w.Type, {})[truck_slug] = truck

        all_tire_variants = {}  # type: Dict[str, List[Any]]
        for slug, tire in all_tires:
            all_tire_variants.setdefault(
                self.get_display_name(tire.UiName),
                [],
            ).append((slug, tire))

        min_friction = Decimal(0)
        max_friction = Decimal(0)
        grouped_tires = (
            set()
        )  # type: Set[Tuple[str, Tuple[Decimal, Decimal, Decimal], bool]]
        for _, tire in all_tires:
            frictions = (
                tire.WheelFriction.BodyFrictionAsphalt,
                tire.WheelFriction.BodyFriction,
                tire.WheelFriction.SubstanceFriction,
            )
            grouped_tires.add(
                (
                    self.get_display_name(tire.UiName),
                    frictions,
                    tire.WheelFriction.IsIgnoreIce,
                )
            )
            max_friction = max(max_friction, *frictions)
            if tire.WheelFriction.IsIgnoreIce:
                min_friction = Decimal(-1)
        max_friction = round_up(max_friction)
        min_friction *= max_friction

        labels = []  # type: List[str]
        datasets = (
            ChartJsDataset(data=[], label="Asphalt"),
            ChartJsDataset(data=[], label="Dirt"),
            ChartJsDataset(data=[], label="Mud"),
        )
        for display_name, frictions, ignore_ice in sorted(
            grouped_tires,
            key=lambda t: (-t[1][2], -t[2], t[0]),
        ):
            labels.append(display_name)
            for dataset, friction in zip(datasets, frictions):
                dataset["data"].append(friction * (-1 if ignore_ice else 1))

        with self.write_data("tires.json") as tires_file:
            self.write_json(
                tires_file,
                {
                    "chartjs": self.tire_chart(
                        labels,
                        datasets,
                        min_friction,
                        max_friction,
                    ),
                    "tires": [
                        {
                            "display_name": display_name,
                            "variants": [
                                self.to_data_tire(slug, v, truck_tires)
                                for slug, v in sorted(variants)
                            ],
                        }
                        for display_name, variants in sorted(all_tire_variants.items())
                    ],
                },
            )

    def to_data_truck(
        self,
        slug: str,
        truck: Truck,
        max_torque: Optional[Decimal] = None,
        max_fuel_consumption: Optional[Decimal] = None,
    ) -> Any:
        min_friction = Decimal(0)
        max_friction = Decimal(0)
        grouped_tires = (
            {}
        )  # type: Dict[Tuple[Decimal, Decimal, Decimal], Dict[bool, Set[str]]]
        for w in truck.CompatibleWheels:
            for t in w.Tires:
                k = (
                    t.WheelFriction.BodyFrictionAsphalt,
                    t.WheelFriction.BodyFriction,
                    t.WheelFriction.SubstanceFriction,
                )
                grouped_tires.setdefault(k, {}).setdefault(
                    t.WheelFriction.IsIgnoreIce, set()
                ).add(self.get_display_name(t.UiName))
                max_friction = max(max_friction, *k)
                if t.WheelFriction.IsIgnoreIce:
                    min_friction = Decimal(-1)
        max_friction = round_up(max_friction)
        min_friction *= max_friction

        tire_labels = []
        frictions = (
            ChartJsDataset(data=[], label="Asphalt"),
            ChartJsDataset(data=[], label="Dirt"),
            ChartJsDataset(data=[], label="Mud"),
        )
        for values, names in sorted(
            grouped_tires.items(),
            key=lambda x: (-x[0][2], -x[0][1], -x[0][0]),
        ):
            for chains in (True, False):
                if chains in names:
                    for i, v in enumerate(values):
                        frictions[i]["data"].append(v * (-1 if chains else 1))
                    tire_labels.append(", ".join(sorted(names[chains])))

        engine_labels = []
        torques = []
        consumption = []
        for es in truck.EngineSocket:
            for e in sorted(es.Engines, key=lambda e: e.Torque):
                engine_labels.append(self.get_display_name(e.UiName))
                torques.append(e.Torque)
                consumption.append(e.FuelConsumption)

        charts = []
        if tire_labels:
            charts.append(
                {
                    "name": "Tires",
                    "chartjs": self.tire_chart(
                        tire_labels,
                        frictions,
                        min_friction,
                        max_friction,
                    ),
                }
            )
        if engine_labels:
            charts.append(
                {
                    "name": "Engines",
                    "chartjs": self.engine_chart(
                        engine_labels,
                        torques,
                        consumption,
                        max_torque=max_torque,
                        max_fuel_consumption=max_fuel_consumption,
                    ),
                }
            )

        return {
            "slug": slug,
            "display_name": self.strings[truck.UiName],
            "charts": charts,
            "python": pprint.pformat(truck),
        }

    def trucks(self, trucks: Dict[str, Truck]) -> None:
        max_torque = Decimal(0)
        max_fuel_consumption = Decimal(0)
        for t in trucks.values():
            for es in t.EngineSocket:
                for e in es.Engines:
                    max_torque = max(max_torque, e.Torque)
                    max_fuel_consumption = max(max_fuel_consumption, e.FuelConsumption)

        with self.write_data("trucks.json") as trucks_file:
            self.write_json(
                trucks_file,
                [
                    self.to_data_truck(
                        name,
                        truck,
                        max_torque=round_up(max_torque),
                        max_fuel_consumption=round_up(max_fuel_consumption),
                    )
                    for name, truck in trucks.items()
                ],
            )


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="Read SnowRunner game data and write to Jekyll _data directory.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase verbosity",
    )
    p.add_argument("-i", "--input", required=True, help="path to initiak.pak")
    p.add_argument("-o", "--output", default=os.path.curdir, help="Jekyll directory")
    args = p.parse_args()

    loglevels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }
    logging.basicConfig(
        stream=sys.stderr,
        level=loglevels.get(args.verbose, logging.DEBUG),
        format="%(name)s %(levelname)-8s :%(lineno)-3d %(message)s",
    )
    # there are lots of "multiple different definitions"
    StringData.logger.setLevel(logging.ERROR + 1)

    data = TruckData.load(args.input)

    jekyll = JekyllOutput(args.output, data.strings)
    jekyll.engines(data.engines, data.trucks)
    jekyll.tires(data.tires, data.trucks)
    jekyll.trucks(data.trucks)

    h = hashlib.sha256()
    with open(args.input, "rb") as fp:
        for b in iter(lambda: fp.read(16384), b""):
            h.update(b)
    with jekyll.write_data("version.json") as version_file:
        jekyll.write_json(
            version_file,
            {
                "date": datetime.now(tz=timezone.utc).isoformat(),
                "hash": h.hexdigest(),
            },
        )


if __name__ == "__main__":
    main()
