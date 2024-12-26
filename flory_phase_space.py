class PhaseSpace:
    def __init__(self, flory_obj, phase_space_data):
        self.flory_obj = flory_obj
        self.chi_matrix = self.flory_obj.chi_matrix
        self.phase_space_data = phase_space_data
        