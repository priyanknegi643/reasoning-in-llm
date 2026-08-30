import torch
import random
import os
import argparse


# if its a sink graph, just reverse the path
def star_or_sink_graph_maker(
    numOfPathsFromSource, lenOfEachPath, maxNodes, shuffle_edge_lists, reverse=False
):
    nodes = list(range(maxNodes))
    random.shuffle(nodes)

    source = nodes.pop()

    edgeList = []
    path = [source]

    for p in range(numOfPathsFromSource):
        oldNode = source
        for i in range(lenOfEachPath - 1):
            newNode = nodes.pop()
            edgeList.append((oldNode, newNode))
            oldNode = newNode
            if p == 0:
                path.append(oldNode)
        if p == 0:
            goal = oldNode

    if shuffle_edge_lists:
        random.shuffle(edgeList)

    if reverse:
        path = path[::-1]

    return edgeList, path, source, goal


def generate_and_save_star_or_sink_graph_data(
    numOfSamples,
    numOfTestSamples,
    numOfPathsFromSource,
    lenOfEachPath,
    maxNodes,
    data_dir,
    generate_test_data=False,
    shuffle_edge_lists=True,
    reverse=False,
    showLoadingBar=True,
):
    os.makedirs(data_dir, exist_ok=True)
    seed = 0
    with open(
        os.path.join(
            data_dir,
            f"graph_{numOfPathsFromSource}_{lenOfEachPath}_sample_{numOfSamples}.txt",
        ),
        "w",
    ) as file:
        for x in range(numOfSamples):
            random.seed(seed)
            seed += 1
            edgeList, path, source, goal = star_or_sink_graph_maker(
                numOfPathsFromSource,
                lenOfEachPath,
                maxNodes,
                shuffle_edge_lists,
                reverse,
            )
            file.write(
                "|".join([",".join([str(i) for i in edge]) for edge in edgeList])
                + f"/{source},{goal}={','.join([str(i) for i in path])}\n"
            )

            # loading bar
            if showLoadingBar:
                numberOfRectangles = int((x + 1) * 50 / numOfSamples)
                bar = "█" * numberOfRectangles + " " * (50 - numberOfRectangles)
                print(f"\r|{bar}| {(x+1)*100/numOfSamples:.1f}%", end="", flush=True)

    if generate_test_data:
        with open(
            os.path.join(
                data_dir,
                f"graph_{numOfPathsFromSource}_{lenOfEachPath}_test_{numOfTestSamples}.txt",
            ),
            "w",
        ) as file:
            for x in range(numOfTestSamples):
                random.seed(seed)
                seed += 1
                edgeList, path, source, goal = star_or_sink_graph_maker(
                    numOfPathsFromSource,
                    lenOfEachPath,
                    maxNodes,
                    shuffle_edge_lists,
                    reverse,
                )
                file.write(
                    "|".join([",".join([str(i) for i in edge]) for edge in edgeList])
                    + f"/{source},{goal}={','.join([str(i) for i in path])}\n"
                )

                # loading bar
                if showLoadingBar:
                    numberOfRectangles = int((x + 1) * 50 / numOfTestSamples)
                    bar = "█" * numberOfRectangles + " " * (50 - numberOfRectangles)
                    print(
                        f"\r|{bar}| {(x+1)*100/numOfTestSamples:.1f}%",
                        end="",
                        flush=True,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate star or sink graph data")
    parser.add_argument(
        "--num_samples", type=int, default=200000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--num_test_samples",
        type=int,
        default=20000,
        help="Number of test samples to generate",
    )
    parser.add_argument(
        "--num_paths", type=int, default=5, help="Number of paths from the source"
    )
    parser.add_argument(
        "--path_length", type=int, default=5, help="Length of each path"
    )
    parser.add_argument(
        "--max_nodes", type=int, default=50, help="Maximum number of nodes in the graph"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/stargraph",
        help="Directory to save the generated data",
    )
    parser.add_argument(
        "--generate_test_data", action="store_true", help="Generate test data"
    )
    args = parser.parse_args()

    generate_and_save_star_or_sink_graph_data(
        numOfSamples=args.num_samples,
        numOfTestSamples=args.num_test_samples,
        numOfPathsFromSource=args.num_paths,
        lenOfEachPath=args.path_length,
        maxNodes=args.max_nodes,
        data_dir=args.data_dir,
        generate_test_data=args.generate_test_data,
    )


# python generate_graph.py --num_samples 500000 --num_paths 2 --path_length 5 --max_nodes 500 --data_dir "data/stargraph"
