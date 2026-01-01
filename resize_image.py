from PIL import Image
import math
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import logging
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('resize_image.log'),
        logging.StreamHandler()
    ]
)

def resize_to_1920(image_path, output_path=None):
    """
    Resize the maximum side of the image to 1920 while maintaining aspect ratio
    """
    try:
        # Open image
        img = Image.open(image_path)
        
        # Get original dimensions
        width, height = img.size
        
        # Calculate scaling ratio
        if width > height:
            # If width is greater than height, use width as reference
            scale = 3840 / width
            new_width = 3840
            new_height = int(height * scale)
        else:
            # If height is greater than or equal to width, use height as reference
            scale = 3840 / height
            new_height = 3840
            new_width = int(width * scale)
        
        # Resize image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Set output path
        if output_path is None:
            output_path = image_path
            
        # Save image
        resized_img.save(output_path)
        
        # Close image
        img.close()
        resized_img.close()
        
        return True, f"Successfully processed {image_path}"
    except Exception as e:
        return False, f"Error processing {image_path}: {str(e)}"

def process_pdf_task(pdf_id):
    """
    Task function to process all images for a single PDF
    """
    try:
        input_dir = f'./data/images_15/{pdf_id}'
        output_dir = f'./data/images_15_3840/{pdf_id}'  # Modified output directory name
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        
        # Sort by page number
        image_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        # Process each image serially
        results = []
        for image_path in tqdm(image_files, desc=f"Processing {pdf_id}", leave=False):
            # Build output path
            rel_path = os.path.relpath(image_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Adjust resolution
            result = resize_to_1920(image_path, output_path)
            results.append(result)
        
        # Count processing results
        success = sum(1 for status, _ in results if status)
        total = len(results)
        
        return True, f"Processed PDF {pdf_id}: {success}/{total} images successful"
    except Exception as e:
        return False, f"Error processing PDF {pdf_id}: {str(e)}"

def clean_output_directory():
    """
    Delete images_1920 directory and all its contents
    """
    output_dir = './data/images_15_3840'  # Modified output directory name
    try:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            logging.info(f"Successfully removed directory: {output_dir}")
    except Exception as e:
        logging.error(f"Error removing directory {output_dir}: {str(e)}")

def main():
    try:
        # Clean output directory
        clean_output_directory()
        
        # Get all PDF IDs
        pdf_ids = os.listdir('./data/images_15')
        
        # Set number of processes
        num_processes = 32
        
        # Use process pool to process PDFs
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(process_pdf_task, pdf_ids),
                total=len(pdf_ids),
                desc="Processing PDFs"
            ))
        
        # Output processing results
        for status, message in results:
            if status:
                logging.info(message)
            else:
                logging.error(message)
                
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")

if __name__ == "__main__":
    main()
