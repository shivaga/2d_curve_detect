from curve2d_detect.read_and_visualize import read_csv, plot_paths
from curve2d_detect.fragment_detection import detect_shapes_in_fragments
from curve2d_detect.visualize_and_save import plot_regularized_shapes, save_as_svg

def main(input_csv, output_svg, occlusion_type='connected', merge_threshold=5):
    paths_XYs = read_csv(input_csv)
    plot_paths(paths_XYs, title="Original Paths")

    # Detect shapes and symmetries in fragmented shapes
    regular_shapes = detect_shapes_in_fragments(paths_XYs, merge_threshold=merge_threshold)
    plot_regularized_shapes(regular_shapes, title="Regularized Shapes")

    # save_as_svg(regular_shapes, file_name=output_svg)
    # print(f"Output saved as {output_svg}")

if __name__ == "__main__":
    input_csv = "./problems/frag0.csv"
    output_svg = "examples/frag0_solution.svg"
    main(input_csv, output_svg)
