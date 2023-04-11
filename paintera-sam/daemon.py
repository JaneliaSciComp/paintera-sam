from segment_anything import SamPredictor, build_sam

import main

def run_as_daemon():
    predictor = SamPredictor(build_sam(checkpoint="/home/caleb/git/saalfeld/paintera-sam/sam_vit_h_4b8939.pth"))
    print("ready!")
    while True:
        args = input().split(" ")
        predictor(args[0], args[1], args[2], predictor)





if __name__ == "__main__":
    run_as_daemon()
