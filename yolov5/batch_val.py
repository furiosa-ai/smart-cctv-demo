import argparse
from collections import OrderedDict
from val import run


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default="data/openimages_person.yaml", help='model.pt path(s)')
    parser.add_argument('--weights', nargs='+', type=str, required=True, help='model.pt path(s)')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[512, 640, 1280], help='model.pt path(s)')

    args = parser.parse_args()

    weights = args.weights
    sizes = args.imgsz

    res = []

    for weight in weights:
        for size in sizes:
            (mp, mr, map50, map, *_), *_ = run(args.data, weights=weight, imgsz=size)

            res.append(OrderedDict([
                ("weights", weight),
                ("size", size),
                ("mp", mp),
                ("mr", mr),
                ("map50", map50),
                ("map", map)
            ]))

    rows = [list(res[0].keys())]

    for exp in res:
        rows.append(list(exp.values()))

    cont = "\n".join(" ".join([str(v) for v in row]) for row in rows)

    with open("res.csv", "w") as f:
        f.write(cont)

    print(cont)


if __name__ == "__main__":
    main()
