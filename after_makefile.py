import util.make_dataset as mk 

def main(num_val):
   mk.make_dataset(int(num_val))

if __name__ == "__main__":
   print("How many dataset to make to validation data")
   main(input())