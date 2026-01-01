import math
from PIL import Image
import os
import shutil
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('merge_image.log'),
        logging.StreamHandler()
    ]
)

def concat_images(image_list, concat_num=1, column_num=2):
    try:
        interval = max(math.ceil(len(image_list) / concat_num), 1)
        concatenated_image_list = list()

        for i in range(0, len(image_list), interval):
            target_dir = "/".join(image_list[0].split("/")[:-1]).replace('images', 'images_50')
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            image_path = target_dir + "/page_{}.png".format(i//interval+1)
            if not os.path.exists(image_path):
                images_this_batch = [
                    Image.open(filename) for filename in image_list[i:i + interval]
                ]
                if column_num==1:
                    total_height = images_this_batch[0].height*len(images_this_batch)
                else:
                    total_height = images_this_batch[0].height*((len(images_this_batch)-1)//column_num+1)

                concatenated_image = Image.new('RGB', (images_this_batch[0].width*column_num, total_height), 'white')
                x_offset, y_offset = 0, 0
                for cnt, image in enumerate(images_this_batch):
                    concatenated_image.paste(image, (x_offset, y_offset))
                    x_offset += image.width
                    if (cnt+1)%column_num==0:
                        y_offset += image.height
                        x_offset = 0
                        image.close()  # Close image immediately to free memory
                concatenated_image.save(image_path)
                concatenated_image.close()
            concatenated_image_list.append(image_path)

        return True, f"Successfully processed images to {target_dir}"
    except Exception as e:
        return False, f"Error processing images: {str(e)}"

def process_pdf(pdf):
    """
    Process all images for a single PDF
    """
    try:
        image_list = os.listdir(os.path.join('./data/images', pdf))
        image_list.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        image_list = [os.path.join('./data/images', pdf, image_path) for image_path in image_list]
        
        if len(image_list) <= 50:
            # Copy files
            target_dir = os.path.join('./data/images_50', pdf)
            os.makedirs(target_dir, exist_ok=True)
            for image_path in image_list:
                shutil.copy(image_path, os.path.join(target_dir, os.path.basename(image_path)))
            return True, f"Copied {pdf} images to {target_dir}"
        else:
            k = math.ceil(len(image_list)/50)
            if 'complong' in pdf:
                if pdf == 'complong_testmini_246':
                    return concat_images(image_list, concat_num=len(image_list)//k, column_num=2)
                else:
                    return concat_images(image_list, concat_num=len(image_list)//k, column_num=1)
            else:
                return concat_images(image_list, concat_num=len(image_list)//k, column_num=k)
    except Exception as e:
        return False, f"Error processing PDF {pdf}: {str(e)}"

def clean_output_directory():
    """
    Delete images_50 directory and all its contents
    """
    output_dir = './data/images_50'
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
        
        # Get all PDFs
        pdf_list = os.listdir('./data/images')
        
        # Set number of processes
        num_processes = 32
        
        # Use process pool to process PDFs
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(process_pdf, pdf_list),
                total=len(pdf_list),
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

if __name__ == '__main__':
    main()

