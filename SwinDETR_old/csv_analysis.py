""" CSV file analysis for car detection """


import csv
import math
import statistics


def distance(pt1, pt2):
    """ euclidean distance between two points(pt1, pt2) """
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


def analyze_single_car(csv_path, dist_threshold=100.0):
    """
    csv_path: 'detections#.csv' file path
    dist_threshold: distance threshold to consider two points as the same car
    """

    cars = []
    # cars = [
    #   {
    #       "centers": [(x_center1, y_center1), (x_center2, y_center2), ...], # center points
    #       "speeds": [v1, v2, ...], # km/h
    #   },
    #   ...
    # ]

    with open(csv_path, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # calculate center point from bounding box
            xmin = float(row['xmin'])
            xmax = float(row['xmax'])
            ymin = float(row['ymin'])
            ymax = float(row['ymax'])
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2

            speed = float(row['km/h']) if row['km/h'].strip() else 0.0
            current_center = (x_center, y_center)

            # (1) check if it is the same car as one of the already registered cars
            found_car_idx = None
            for i, car in enumerate(cars):
                # check if the current center is close to the last center of the car
                last_center = car["centers"][-1]
                if distance(current_center, last_center) < dist_threshold:
                    found_car_idx = i
                    break

            # (2) if close to any car -> update the car
            if found_car_idx is not None:
                cars[found_car_idx]["centers"].append(current_center)
                cars[found_car_idx]["speeds"].append(speed)
            else:
                # (3) if not close to any car -> register a new car
                cars.append({
                    "centers": [current_center],
                    "speeds": [speed]
                })

    # now, cars list contains multiple "physical cars" separated
    total_cars = len(cars)

    # average speed of all cars
    all_speeds = []
    for car in cars:
        all_speeds.extend(car["speeds"])

    avg_speed = statistics.mean(all_speeds) if all_speeds else 0.0

    print(f"[Analysis Result]")
    print(f"- Total Number of Cars: {total_cars}")
    print(f"- Average Speed: {avg_speed:.2f}km/h (distance threshold={dist_threshold})")
    return total_cars, avg_speed


if __name__ == "__main__":
    csv_path = "./record/detections4.csv"
    cars_count, avg_speed = analyze_single_car(csv_path, dist_threshold=50.0)