import numpy as np
import tifffile
import psutil
import os
import time

def generate_test_tiff(output_path, target_size_gb=4, chunk_shape=(2048, 2048)):
    """
    Generates a test BigTIFF file with 16-bit data
    """
    process = psutil.Process(os.getpid())
    dtype = np.uint16
    bytes_per_pixel = np.dtype(dtype).itemsize
    pixels_per_chunk = chunk_shape[0] * chunk_shape[1]
    chunk_size_bytes = pixels_per_chunk * bytes_per_pixel
    target_size_bytes = target_size_gb * 1024**3
    num_chunks = int(target_size_bytes // chunk_size_bytes)

    memory_log = []
    start_time = time.time()

    with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
        # Precompute one chunk pattern (gradient)
        chunk_pattern = np.arange(pixels_per_chunk, dtype=dtype)
        chunk_pattern = (chunk_pattern % 65535).reshape(chunk_shape)
        
        for i in range(num_chunks):
            tif.write(chunk_pattern)
            
            # Log memory every 10 chunks
            if i % 10 == 0:
                mem_mb = process.memory_info().rss / 1024**2
                memory_log.append((time.time() - start_time, mem_mb))
                print(f"Progress: {i+1}/{num_chunks} chunks | "
                      f"Memory: {mem_mb:.1f} MB", end='\r')

    print(f"\nCreated test TIFF: {os.path.getsize(output_path)/1024**3:.2f} GB")
    return memory_log

def process_with_logging(input_path, output_path):
    """
    Processes TIFF with memory logging
    """
    process = psutil.Process(os.getpid())
    memory_log = []
    start_time = time.time()

    with tifffile.TiffFile(input_path) as tif:
        with tifffile.TiffWriter(output_path, bigtiff=True) as tif_writer:
            for i, page in enumerate(tif.pages):
                img = page.asarray(out='memmap')  # Memory-mapped
                img_8bit = (img // 256).astype(np.uint8)  # Simple scaling
                tif_writer.write(img_8bit)
                
                # Log memory every 10 pages
                if i % 10 == 0:
                    mem_mb = process.memory_info().rss / 1024**2
                    memory_log.append((time.time() - start_time, mem_mb))
                    print(f"Processed {i+1} pages | "
                          f"Memory: {mem_mb:.1f} MB", end='\r')

    print(f"\nCreated output TIFF: {os.path.getsize(output_path)/1024**3:.2f} GB")
    return memory_log

if __name__ == "__main__":
    gen_log = generate_test_tiff("test_4gb.tif", target_size_gb=16.0)  # Start with 0.1 GB (100MB)
    proc_log = process_with_logging("test_4gb.tif", "output_8bit.tif")
    print(proc_log) 
