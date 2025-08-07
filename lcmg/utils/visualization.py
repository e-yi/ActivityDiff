# import io
from typing import List

# import cairosvg
from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog('rdApp.*')


def mol2img(mol: Chem.Mol, *, size=(300, 300), dpi=96, show_2d=True, show_idx=True, label=None):
    """
    Convert an RDKit molecule to an image.

    :param mol: The input RDKit molecule to be converted to an image.
    :param size: The size of the output image (width, height).
    :param dpi: dpi, disabled temporarily
    :param show_2d: Whether to show a 2D representation of the molecule.
    :param show_idx: Whether to label atoms with their indices.
    :param label: Optional label to add to the image, positioned at the bottom.

    :return: An image of the molecule with optional labeling.
    """

    mol = Chem.Mol(mol)

    if show_2d:
        mol.RemoveAllConformers()

    if show_idx:
        for idx in range(mol.GetNumAtoms()):
            mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(idx))

    # width, height = size
    # drawer = rdMolDraw2D.MolDraw2DSVG(width=width,height=height)
    # drawer.DrawMolecule(mol)
    # drawer.FinishDrawing()
    # png = cairosvg.svg2png(bytestring=drawer.GetDrawingText().encode(), dpi=dpi)
    # img = Image.open(io.BytesIO(png))
    img = Chem.Draw.MolToImage(mol, size=size)

    if label is not None:
        draw = ImageDraw.Draw(img)
        draw.text((5, size[1] - 20), label, fill=(0, 0, 0), font_size=10)

    return img


def mol2grid_img(mols_list, *, size=(200, 200), img_per_row=3, show_2d=True, show_idx=False, label_per_img=None,
                 label_grid=None):
    """
    Create a grid image displaying multiple RDKit molecules.

    :param mols_list: List of RDKit molecules to display in the grid.
    :param size: Size of each individual molecule image.
    :param img_per_row: Number of images to display per row in the grid.
    :param show_2d: Whether to show 2D representations of the molecules.
    :param show_idx: Whether to label atoms with their indices.
    :param label_per_img: Labels for each individual image; can be a string or list of strings.
    :param label_grid: Optional label for the entire grid, positioned at the bottom.

    :return: A combined image of all molecules arranged in a grid format.
    """
    num_mols = len(mols_list)

    # Calculate the number of rows needed
    num_rows = (num_mols + img_per_row - 1) // img_per_row

    x = img_per_row * size[0]
    y = num_rows * size[1] + (20 if label_grid is not None else 0)
    # Create a new image to display all molecules
    combined_image = Image.new('RGB', (x, y), (255, 255, 255))

    if label_per_img is None or isinstance(label_per_img, str):
        label_per_img = [label_per_img] * num_mols

    # Paste each molecule into the combined image
    for i, (mol, label_per_img) in enumerate(zip(mols_list, label_per_img, strict=True)):
        mol = Chem.Mol(mol)
        col_index = i % img_per_row
        row_index = i // img_per_row

        img = mol2img(mol, size=size, show_2d=show_2d, show_idx=show_idx, label=label_per_img)

        combined_image.paste(img, (col_index * size[0], row_index * size[1]))

    if label_grid is not None:
        draw = ImageDraw.Draw(combined_image)
        draw.text((20, y - 20), label_grid, fill=(0, 0, 0), font_size=10)

    # Display the combined image
    return combined_image


def frames2gif(frames: List[Image.Image], output_path, *, duration=100, last_frame_duration=6000, ):
    """
    Create a GIF from a list of image frames.

    :param frames: List of images to include in the GIF.
    :param output_path: Path to save the generated GIF.
    :param duration: Duration for each frame in milliseconds.
    :param last_frame_duration: Duration for the last frame in milliseconds.

    :return: None, saves the GIF to the specified output path.
    """
    # Set durations for each frame
    for img in frames[:-1]:
        img.info['duration'] = duration
    frames[-1].info['duration'] = last_frame_duration
    frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0)


def diff2gif(diff_dict, output_path, *, label=None, duration=100, last_frame_duration=6000,
             size=(200, 200), img_per_row=3, select_samples=slice(None), show_2d=False):
    """
    Generate a GIF from simulation results containing multiple RDKit molecules over time.

    :param diff_dict: Dictionary where keys are timesteps and values are lists of RDKit molecules.
    :param output_path: Path to save the generated GIF.
    :param label: Optional label for each frame.
    :param duration: Duration for each frame in milliseconds.
    :param last_frame_duration: Duration for the last frame in milliseconds.
    :param size: Size of each individual molecule image.
    :param img_per_row: Number of images to display per row in the grid.
    :param select_samples: Slice to select specific samples from the list of molecules.

    :return: None, saves the GIF to the specified output path.
    """
    timesteps = sorted(diff_dict.keys(), reverse=True)
    T = timesteps[0]

    frames = []
    for t in timesteps:
        mols = diff_dict[t][select_samples]
        img = mol2grid_img(mols, size=size, img_per_row=img_per_row, show_2d=show_2d, show_idx=False,
                           label_per_img=f'{t}/{T}', label_grid=label)
        frames.append(img)

    frames2gif(frames, output_path, duration=duration, last_frame_duration=last_frame_duration)
