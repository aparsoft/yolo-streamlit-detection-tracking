# Video attribution

The sample videos in this folder come from [Pexels](https://www.pexels.com/) and are used under the
[Pexels licence](https://www.pexels.com/license/): free to use, including commercially, and to modify, with no
attribution required. We credit the creators anyway.

Each is the first 20 seconds of the original, re-encoded to 1280 px wide at 30 fps with no audio, to keep the
repo small. They were chosen on Oct 7, 2026 by running this app's own detector and tracker (yolo26n + ByteTrack, the
app's defaults) over 731 candidate clips and keeping the ones with a steady camera, confident detections and stable
track IDs, where people are small enough that faces aren't the subject.

| File | Shows | Original | Creator | Licence |
|---|---|---|---|---|
| `pedestrians_summer_street.mp4` | People counting and tracking (the default) | [Pedestrians Cross Busy City Street on Summer Day](https://www.pexels.com/video/pedestrians-cross-busy-city-street-on-summer-day-38045690/) | K | Pexels |
| `crosswalk_red_car.mp4` | YOLO World: "red car"; people, a bus | [People Waiting To Cross The Street On The Pedestrian Lane](https://www.pexels.com/video/people-waiting-to-cross-the-street-on-the-pedestrian-lane-2836277/) | K | Pexels |
| `street_person_in_red.mp4` | YOLO World: "person in red"; a bicycle | [Lively Urban Scene with Diverse Pedestrians](https://www.pexels.com/video/lively-urban-scene-with-diverse-pedestrians-37567255/) | Image Hunter | Pexels |
| `street_dogs_bikes.mp4` | Dogs, bicycles and people in one scene | [Vibrant City Street with People Strolling](https://www.pexels.com/video/vibrant-city-street-with-people-strolling-30507204/) | Evgenij Mikhailov | Pexels |
| `highway_bridge_traffic.mp4` | Vehicle tracking and counting | [Dynamic City Traffic on Elevated Highway Bridge](https://www.pexels.com/video/dynamic-city-traffic-on-elevated-highway-bridge-37494789/) | Aswin R S | Pexels |
| `city_traffic_dense.mp4` | Dense traffic: many IDs at once | [Traffic in a City](https://www.pexels.com/video/traffic-in-a-city-5681670/) | Hervé Piglowski | Pexels |
| `yoga_warrior_pose.mp4` | Pose estimation | [People Doing the Warrior II Pose at a Yoga Class](https://www.pexels.com/video/people-doing-the-warrior-ii-pose-at-a-yoga-class-8480549/) | Yan Krukau | Pexels |
| `stretching_from_above.mp4` | Pose estimation, seen from above | [Cheerleaders Stretching Together](https://www.pexels.com/video/cheerleaders-stretching-together-7894189/) | MART PRODUCTION | Pexels |
| `busy_intersection.mp4` | Paths crossing: ReID trackers, multi-video | [Busy City Intersection with Pedestrians Crossing](https://www.pexels.com/video/busy-city-intersection-with-pedestrians-crossing-33865608/) | SHOX ART | Pexels |

Older clips are in `archive/`, which the app's video picker doesn't scan.
