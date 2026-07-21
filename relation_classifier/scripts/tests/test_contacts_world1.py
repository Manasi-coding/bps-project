"""
test_contacts_world1.py

PURPOSE: answer one narrow question before we build the full 6-world
extract_gazebo_contacts.py pipeline -- does Gazebo Harmonic's physics
engine actually generate contact-sensor output for two OBJECTS THAT ARE
BOTH <static>true</static>? Many physics engines (DART/ODE/Bullet) skip
collision resolution entirely for static-static pairs as an optimization,
which would mean these sensors report nothing regardless of whether the
objects are touching. This script is small and disposable -- it launches
ONE modified world (world1_baseline.sdf, with contact sensors already
added), subscribes to a couple of specific contact topics, and just
prints whatever comes back.

RUN THIS ON YOUR MACHINE (not in this sandbox -- no gz install/network
here). Requires: gz-sim Harmonic on PATH, gz.transport13 + gz.msgs10
Python bindings importable.

HOW TO READ THE RESULT:
  - If you see "contact" messages with nonzero force/points for a pair
    that the SDF geometry heuristic already calls Contact/Support
    (obj_bottle/obj_mug are ~8.5cm apart center-to-center per the SDF,
    both radius ~3cm -- check whether that's within touching tolerance
    for your EPSILON_M) -> static-static contact sensing works, proceed
    with the full pipeline as originally planned (no <static> change
    needed).
  - If NOTHING publishes on either topic within the timeout -> static-
    static contact sensing is not happening in your engine config, and
    we need to revisit (flip obj_* to <static>false>, or find another
    signal). Also try `gz topic -l | grep contact` in another terminal
    while this is running -- if the topic doesn't even show up, the
    sensor isn't attaching to the model the way we assumed and the
    topic-naming guess in this script is wrong (report back what you
    see so we fix it before writing the real script).

Nothing here is a replacement for the SDFGeometryRelationSource heuristic
-- it's purely a go/no-go check for the physics-based cross-check idea.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

WORLD_SDF = Path("../../../worlds_with_contact_sensors/world1_baseline.sdf")  # the MODIFIED file from worlds_with_contact_sensors.zip
WORLD_NAME = "bps_room"                     # the SDF's internal <world name='...'>, NOT the filename stem

# A couple of pairs to check -- pick names actually close together in
# world1_baseline per the SDF poses, so a positive result is plausible.
OBJECTS_TO_WATCH = ["obj_bottle", "obj_mug", "obj_bowl", "obj_apple"]
LINK_NAME = "link"  # confirmed uniform across all obj_* models in this project

# Best-guess default gz-sim scoped-topic pattern for a sensor with no
# explicit <topic> override in its SDF. VERIFY this against `gz topic -l`
# once world1 is running -- if it's wrong, no messages will ever arrive
# even if the sensor itself is working, and that's a red herring for the
# "does static-static contact work" question. Don't conclude "no contact"
# from silence here without checking `gz topic -l` first.
TOPIC_TEMPLATE = "/world/{world}/model/{model}/link/{link}/sensor/{model}_contact_sensor/contact"

SETTLE_STEPS = 50   # placeholder; not used for timing in this real-time test,
                    # kept here as a reminder this is the same constant the
                    # full pipeline will use
TIMEOUT_S = 15.0


def main() -> int:
    try:
        from gz.transport13 import Node
        from gz.msgs10.contacts_pb2 import Contacts
    except ImportError as e:
        print(f"Cannot import gz.transport13 / gz.msgs10: {e}")
        print("This must be run on a machine with Gazebo Harmonic's Python bindings installed.")
        return 1

    if not WORLD_SDF.exists():
        print(f"World file not found: {WORLD_SDF.resolve()}")
        print("Point WORLD_SDF at the MODIFIED world1_baseline.sdf (with contact sensors added).")
        return 1

    print(f"Launching `gz sim -s -r {WORLD_SDF}` ...")
    proc = subprocess.Popen(["gz", "sim", "-s", "-r", str(WORLD_SDF)])

    received: dict[str, list] = {name: [] for name in OBJECTS_TO_WATCH}

    def make_callback(object_name: str):
        def callback(msg: Contacts):
            received[object_name].append(msg)
            print(f"  [{object_name}] contact message: {len(msg.contact)} contact(s)")
            for c in msg.contact:
                print(f"      {c.collision1.name} <-> {c.collision2.name}, "
                      f"{len(c.position)} contact point(s)")
        return callback

    node = Node()
    topics = {}
    for name in OBJECTS_TO_WATCH:
        topic = TOPIC_TEMPLATE.format(world=WORLD_NAME, model=name, link=LINK_NAME)
        topics[name] = topic
        ok = node.subscribe(Contacts, topic, make_callback(name))
        print(f"Subscribed to {topic}: {'ok' if ok else 'FAILED'}")

    print(f"\nLetting the world run for {TIMEOUT_S:.0f}s (real-time) to see what publishes...")
    try:
        time.sleep(TIMEOUT_S)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    any_received = False
    for name in OBJECTS_TO_WATCH:
        n = len(received[name])
        print(f"  {name:15s} ({topics[name]}): {n} message(s) received")
        if n > 0:
            any_received = True

    if any_received:
        print("\n-> Static-static contact sensing APPEARS to work. Safe to proceed "
              "with the full pipeline without changing <static>.")
    else:
        print("\n-> NOTHING was received on any watched topic within the timeout.")
        print("   Before concluding static-static contacts don't work, run "
              "`gz topic -l | grep contact` while this script is running (in a "
              "separate terminal, shorten TIMEOUT_S or add a pause) to confirm "
              "the topics actually exist under the names above. If they don't "
              "exist at all, the topic-naming guess in this script is wrong -- "
              "report the actual topic list back and we'll fix the pipeline "
              "script accordingly. If the topics DO exist but never publish, "
              "that's the static-static collision-pruning behavior, and we "
              "should revisit the <static> flag.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
