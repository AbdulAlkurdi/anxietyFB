from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
startTime = datetime.now()

def process_item(item):
    # Your processing code goes here
    #print(f"Processing {item}")
    # Replace the following line with your actual processing code
    return item * item

# Example usage of ProcessPoolExecutor to parallelize a for loop
with ProcessPoolExecutor() as executor:
    # Submit tasks to the process pool
    
    future_to_item = {executor.submit(process_item, item): item for item in range(100000)}
    
    # Process as tasks complete
    for future in as_completed(future_to_item):
        item = future_to_item[future]
        try:
            result = future.result()
            # Do something with the result if needed
            #print(f"Result of item {item}: {result}")
        except Exception as exc:
            print(f"Item {item} generated an exception: {exc}")

print('total time for parallel ',datetime.now() - startTime)


startTime = datetime.now()

for i in range(100000):
    result2 = i*i

print('total time for loop',datetime.now() - startTime)
