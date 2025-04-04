from NPProblem import Denoise, Generate
import click


@click.command()
@click.argument(
    "source",
    type=click.Path(exists=True)
    )
@click.argument(
    "target",
    type=click.Path()
    )
@click.option(
    "--problem", "-p",
    default=None,
    type=click.Choice(["Denoise", "Generate"]),
    help="Select a NP problem as the source"
    )
@click.option(
    "--graph-type", "-g",
    default="king",
    type=click.Choice(["nearest", "king"]),
    help="Graph type (topology) of the spins"
    )
def main(
    source, target, problem, graph_type
    ):
    match problem:
        case "Denoise":
            Denoise.convert(source, target)
        case "Generate":
            Generate.convert(source, target, graph_type)
        case _:
            raise ValueError("Invalid Problem")


if __name__ == "__main__":
    main()
