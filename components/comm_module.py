import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from utils.dotdic import DotDic
from utils.invfunc import InverseFunc


class CommModule:
    def __init__(self, args):
        self.args = args

        self.loc_UE = self.gen_location_UE()
        self.loc_BS = self.gen_location_BS()
        self.lsfc0_dB = self.gen_lsfc_dB_no_sf()
        self.omega_tilde = self.cal_omega_tilde()
        self.pilots = self._gen_non_orth_pilots()

    def random_drop_hexagon(self, n_drops, i_drop):
        args = self.args

        np.random.seed(args.random_seed_loc + i_drop)
        temp = np.random.rand(n_drops)
        n_rhombus1 = np.sum(temp < 1/3)
        n_rhombus2 = np.sum(temp >= 2/3)
        n_rhombus3 = n_drops - (n_rhombus1 + n_rhombus2)

        u1 = np.random.rand(n_rhombus1)
        v1 = np.random.rand(n_rhombus1)
        u2 = np.random.rand(n_rhombus2)
        v2 = np.random.rand(n_rhombus2)
        u3 = np.random.rand(n_rhombus3)
        v3 = np.random.rand(n_rhombus3)

        loc = np.zeros((n_drops, 2))
        loc[:n_rhombus1, 0] = np.sqrt(3)/2 * u1
        loc[:n_rhombus1, 1] = -1/2*u1 + v1
        loc[n_rhombus1:-n_rhombus3, 0] = -np.sqrt(3)/2 * (u2 - v2)
        loc[n_rhombus1:-n_rhombus3, 1] = -1/2 * (u2 + v2)
        loc[-n_rhombus3:, 0] = -np.sqrt(3)/2 * v3
        loc[-n_rhombus3:, 1] = u3 - 1/2*v3

        loc = loc * args.r_cell

        return loc

    def gen_location_UE(self):
        args = self.args

        i_drop = 0

        loc = self.random_drop_hexagon(args.n_agents, i_drop)

        while True:
            dist = np.sqrt(loc[:, 0] ** 2 + loc[:, 1] ** 2)
            if np.min(dist) >= args.exclusion_zone:
                break

            i_drop += 1
            ind_redrop = np.where(dist < args.exclusion_zone)[0]
            n_redrop = len(ind_redrop)

            loc[ind_redrop] = self.random_drop_hexagon(n_redrop, i_drop)

        return loc

    def gen_location_BS(self):
        args = self.args

        angle = np.linspace(1/6, 11/6, 6) * np.pi

        loc = np.zeros((7, 2))
        loc[1:, 0] = np.cos(angle) * np.sqrt(3) * args.r_cell
        loc[1:, 1] = np.sin(angle) * np.sqrt(3) * args.r_cell

        return loc

    def gen_lsfc_dB_no_sf(self):
        args = self.args

        dist = np.sqrt((self.loc_UE[:, 0] - self.loc_BS[:, 0].reshape(-1, 1))**2
                       + (self.loc_UE[:, 1] - self.loc_BS[:, 1].reshape(-1, 1))**2)

        lsfc0_dB = -140.6 - 37.6 * np.log10(dist * 1e-3)
        lsfc0_dB -= args.noise_power_dB

        return lsfc0_dB

    def gen_lsfc_dB(self):
        args = self.args

        lsfc_dB = self.lsfc0_dB + args.noise_power_dB
        lsfc_dB = lsfc_dB + np.random.randn(7, args.n_agents) * args.shadow_fading_std
        lsfc_dB = np.max(lsfc_dB, axis=0)
        lsfc_dB -= args.noise_power_dB

        return lsfc_dB

    def cal_omega_tilde(self):
        args = self.args

        inv_v_func = InverseFunc(self._v_func)
        ub = inv_v_func.cal(norm.isf(args.error_prob) / np.sqrt(args.block_len - args.n_pilots))

        inv_omega_func = InverseFunc(self._omega_func)

        omega_tilde = np.zeros(args.n_agents)
        for i in range(args.n_agents):
            arg = args.rate_thresh[i] * args.block_len * np.log(2) / (args.block_len - args.n_pilots)
            omega_tilde[i] = inv_omega_func.cal(arg, ub=ub)

        return omega_tilde

    def _v_func(self, x):
        v = (x + 1) / (np.sqrt(2*x + 1)) * np.log(1 + 1/x)
        return v

    def _omega_func(self, x):
        args = self.args

        omega = np.log(1 + 1/x) - (norm.isf(args.error_prob) / np.sqrt(args.block_len - args.n_pilots)
                                   * np.sqrt(2*x + 1) / (x + 1))

        return omega

    def _gen_non_ortho_pilots(self):
        args = self.args

        pilots = (np.random.randn(args.pilot_len, args.n_pilots) +
                  1j * np.random.randn(args.pilot_len, args.n_pilots)) / np.sqrt(2)

        pilots = pilots / np.linalg.norm(pilots, axis=0)

        return pilots

    def cal_C_mat(self, active_pilots, lsfc):

        rho0 = args.pilot_len * np.min(lsfc) * args.max_power

        temp = self.pilots[:, active_pilots]
        cov_mat = np.eye(len(active_pilots)) - np.linalg.inv(rho0 * temp.T.conj() @ temp + np.eye(len(active_pilots)))

        return cov_mat

    def cal_C_tilde_mat(self, backlogged_users, lsfc):
        active_pilots = backlogged_users

        rho0 = args.pilot_len * np.min(lsfc) * args.max_power
        


    def cal_intrf_tol_mr(self, lsfc):
        args = self.args

        intrf_tol = ((args.n_antennas - 1) * args.n_pilots * np.min(lsfc) * args.max_power * self.omega_tilde - 1) / \
                    (1 + args.n_pilots * np.min(lsfc) * args.max_power) * lsfc

        return intrf_tol

    def cal_intrf_tol_zf(self, lsfc, n_active_pilots):
        args = self.args

        intrf_tol = ((args.n_antennas - n_active_pilots) * args.n_pilots * np.min(lsfc) * args.max_power * self.omega_tilde - 1) / \
                    (1 + args.n_pilots * np.min(lsfc) * args.max_power) * lsfc

        return intrf_tol


if __name__ == '__main__':
    args = DotDic()
    args.r_cell = 1000
    args.random_seed_loc = 10086
    args.n_agents = 20
    args.exclusion_zone = 50
    args.bandwidth = 0.18e+6
    args.noise_floor = -169
    args.noise_figure = 0
    args.noise_power_dB = (args.noise_floor-30) + 10*np.log10(args.bandwidth) + args.noise_figure
    args.shadow_fading_std = 8
    args.error_prob = 1e-5
    args.block_len = 20
    args.n_pilots = 6
    args.rate_thresh = np.array([1] * args.n_agents)

    comm = CommModule(args)
    loc = comm.random_drop_hexagon(1000, 1)
    plt.scatter(loc[:, 0], loc[:, 1])
    plt.show()

    plt.scatter(comm.loc_UE[:, 0], comm.loc_UE[:, 1], marker='x')
    plt.scatter(comm.loc_BS[:, 0], comm.loc_BS[:, 1], marker='^')
    plt.show()

    plt.stem(range(args.n_agents), comm.lsfc0_dB[0, :])
    plt.show()

    plt.stem(range(args.n_agents), comm.gen_lsfc_dB())
    plt.show()

    plt.stem(range(args.n_agents), comm.omega_tilde)
    plt.show()
