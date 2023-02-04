colors = torch.tensor([
        [0, 0, 0], # 0 background
        [255, 0, 0], # 1 dirt
        [255, 85, 0], # 2 airplane
        [255, 170, 0], # 3 mage
        [255, 255, 0], # 4 dog
        [170, 255, 0], # 5 bird
        [85, 255, 0], # 6 fish
        [0, 255, 0], # 7 scientist
        [0, 255, 85], # 8 thief
        [0, 255, 170], # 9 ship
        [0, 255, 255], # 10 ball
        [0, 170, 255], # 11 robot
        [0, 85, 255], # 12 queen
        [0, 0, 255], # 13 sword
        [85, 0, 255], # 14 wall
        [170, 0, 255], # 15 no_message
        [255, 0, 255], # 16 with_message
    ])

    

class Stats:
    def __init__(self, eval_length, vis_length):
        self.real_step_count = 0
        self.imag_step_count = 0
        self.real_loss_total = 0
        self.imag_loss_total = 0

        self.real_tp = torch.zeros(17, dtype=int, device=device)
        self.real_fn = torch.zeros(17, dtype=int, device=device)
        self.real_fp = torch.zeros(17, dtype=int, device=device)
        self.real_tn = torch.zeros(17, dtype=int, device=device)

        self.imag_tp = torch.zeros(17, dtype=int, device=device)
        self.imag_fn = torch.zeros(17, dtype=int, device=device)
        self.imag_fp = torch.zeros(17, dtype=int, device=device)
        self.imag_tn = torch.zeros(17, dtype=int, device=device)

        self.real_dists = []
        self.imag_dists = []