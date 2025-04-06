from NPProblem import Denoise, Generate
import click


@click.command()
@click.argument(
    "source",
    type=click.Path(exists=True)
    )
@click.option(
    "--problem", "-p",
    default=None,
    type=click.Choice(["Generate", "Denoise"]),
    help="Select a NP problem as the source"
    )
@click.option(
    "--graph-type", "-g",
    default="king",
    type=click.Choice(["nearest", "king"]),
    help="Graph type (topology) of the spins"
    )
@click.option(
    "--color-type", "-c",
    default="grayscale",
    type=click.Choice(["monochrome", "grayscale"]),
    help="Color type (space) of the spins"
    )
def main(
    source, problem, graph_type, color_type
    ):
    match problem:
        case "Generate":
            Generate.convert(source, graph_type)
        case "Denoise":
            Denoise.convert(source, color_type)
        case _:
            raise ValueError("Invalid Problem")


if __name__ == "__main__":
    main()
