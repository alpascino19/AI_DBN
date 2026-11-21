"""Punct de intrare: python -m vigil_vision [--fake]. [CV2]

    python -m vigil_vision                 # camera reală -> vigil/vision
    python -m vigil_vision --show          # + fereastră de debug (zonă piept, telefon, evenimente)
    python -m vigil_vision --no-mqtt       # doar consolă
    python -m vigil_vision --fake          # date false (scenariul demo), fără cameră
    python -m vigil_vision --fake --scenario chest
"""
from __future__ import annotations

import argparse
import logging

from .config import load_config
from .publisher import Publisher


def main() -> None:
    ap = argparse.ArgumentParser(prog="vigil_vision", description="Vigil - pipeline de viziune")
    ap.add_argument("--config", help="cale config.yaml (implicit Vision/config.yaml)")
    ap.add_argument("--fake", action="store_true", help="publică date false, fără cameră")
    ap.add_argument("--scenario", default="demo",
                    help="cu --fake: demo | normal | phone | drowsy | chest | unconscious")
    ap.add_argument("--camera", type=int, help="indexul camerei")
    ap.add_argument("--show", action="store_true", help="fereastră de debug (doar local)")
    ap.add_argument("--no-mqtt", action="store_true", help="nu te conecta la broker")
    ap.add_argument("--echo", action="store_true", help="afișează în consolă și când MQTT merge")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    if args.camera is not None:
        cfg["camera"]["index"] = args.camera
    pub = Publisher(cfg, enabled=not args.no_mqtt, echo=args.echo)

    if args.fake:
        from .fake.fake_vision import run_fake
        run_fake(cfg, pub, scenario=args.scenario)
        return

    from .pipeline import Pipeline
    Pipeline(cfg, publisher=pub, show=args.show).run()


if __name__ == "__main__":
    main()
