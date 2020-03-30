import numpy as np
from multiprocessing import Pool
from itertools import product
import os

try:
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    from plotly.offline import init_notebook_mode, iplot
except ImportError:
    print("Installing plotly. This may take a while.")
    from pip._internal import main as pipmain
    pipmain(['install', 'plotly'])
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    from plotly.offline import init_notebook_mode, iplot

class corona_model(object):

    def __init__(self, ξ_base, A_rel, d_vaccine, rel_ρ, δ_param, \
                 ωR_param, π_D, R_0, rel_λ,initial_infect):
        # "I want an instance of this model, with these parameters"

        self.pop        = 340_000_000
        self.T_years    = 5
        self.Δ_time     = 14
        self.T          = self.T_years * 365 * self.Δ_time
        self.λ          = 1
        self.γ          = 0
        self.ξ_base_high= .999
        self.r_high     = .999
        self.r          = .98
        self.δ          = 1/(self.Δ_time*δ_param)
        self.ωR         = 1/(self.Δ_time*ωR_param)

        self.ωD         = self.ωR*π_D/(1-π_D)
        self.λQ         = rel_λ*self.λ
        self.ρS         = R_0/((self.λ/self.δ)*(rel_ρ + self.δ/(self.ωR+self.ωD)))
        self.ρA         = rel_ρ*self.ρS

        self.InitialInfect = initial_infect
        self.d_vaccine     = d_vaccine
        self.A_rel         = A_rel
        self.ξ_base        = ξ_base

        self.baseline = {
            'τA'            : 0.,
            'ξ_U'           : 0.,
            'ξ_P'           : 0.,
            'ξ_N'           : 0.,
            'ξ_R'           : 0.,
            'r_U'           : self.r,
            'r_P'           : 0.,
            'r_N'           : self.r,
            'r_R'           : self.r_high,
            'd_start_exp'   : 0.,
            'experiment'    : "baseline_vaccine_tag"
        }

        τ_A_daily_target    = 0
        ξ_U_daily_target	= ξ_base
        ξ_P_daily_target	= self.ξ_base_high
        ξ_N_daily_target	= ξ_base
        ξ_R_daily_target	= 0

        r_U_daily_target	= 0
        r_N_daily_target	= 0
        r_P_daily_target	= 0
        r_R_daily_target	= self.r_high

        self.policy_offset = 14

        self.common_quarantine = {
            'τA'            : (1+τ_A_daily_target)**(1./self.Δ_time)-1,
            'ξ_U'           : (1+ξ_U_daily_target)**(1./self.Δ_time)-1,
            'ξ_P'           : (1+ξ_P_daily_target)**(1./self.Δ_time)-1,
            'ξ_N'           : (1+ξ_N_daily_target)**(1./self.Δ_time)-1,
            'ξ_R'           : (1+ξ_R_daily_target)**(1./self.Δ_time)-1,
            'r_U'           : (1+r_U_daily_target)**(1./self.Δ_time)-1,
            'r_P'           : (1+r_P_daily_target)**(1./self.Δ_time)-1,
            'r_N'           : (1+r_N_daily_target)**(1./self.Δ_time)-1,
            'r_R'           : (1+r_R_daily_target)**(1./self.Δ_time)-1,
            'experiment'    : "baseline_vaccine_tag"
        }

    def solve_case(self, model):
        # Model is {baseline,commonquarantine,testquarantine}
        M0_vec = np.zeros(13)
        M0_vec[4] = self.InitialInfect / self.pop
        M0_vec[8] = 1. / self.pop
        M0_vec[0] = 1 - np.sum(M0_vec)

        Q_inds      = [1,3,5,7,9,11]
        NQ_inds     = [0,2,4,6,8,10]
        IANQ_inds   = [4,6]
        IAQ_inds    = [5,7]
        ISNQ_inds   = [8]
        ISQ_inds    = [9]
        NANQ_inds   = [0,2]
        RANQ_inds   = [10]
        NAQ_inds    = [1,3]
        RAQ_inds    = [11]

        M_t = np.zeros((13, self.T))
        M_t[:,0] = M0_vec

        for t in range(1,self.T):
            Mt = M_t[:,t-1]

            Mt_Q        = np.sum(Mt[Q_inds])
            Mt_NQ       = np.sum(Mt[NQ_inds])

            Mt_IANQ     = np.sum(Mt[IANQ_inds])
            Mt_IAQ      = np.sum(Mt[IAQ_inds])

            Mt_ISNQ     = np.sum(Mt[ISNQ_inds])
            Mt_ISQ      = np.sum(Mt[ISQ_inds])

            Mt_NANQ     = np.sum(Mt[NANQ_inds])
            Mt_RANQ     = np.sum(Mt[RANQ_inds])

            Mt_NAQ      = np.sum(Mt[NAQ_inds])
            Mt_RAQ      = np.sum(Mt[RAQ_inds])

            Mt_Total    = self.λ*Mt_NQ + self.λQ*Mt_Q
            Mt_I        = self.λ*(Mt_IANQ + Mt_ISNQ) + self.λQ*(Mt_IAQ + Mt_ISQ)
            Mt_N        = self.λ*(Mt_NANQ + Mt_RANQ) + self.λQ*(Mt_NAQ + Mt_RAQ)

            pit_I       = Mt_I/Mt_Total
            pit_IA      = (self.λ*Mt_IANQ + self.λQ*Mt_IAQ)/Mt_I
            pit_IS      = (self.λ*Mt_ISNQ + self.λQ*Mt_ISQ)/Mt_I

            alphat      = pit_I*(pit_IS*self.ρS + pit_IA*self.ρA)

            # A_daily just selects every 14th entry starting at the 14th entry (end of day each day)

            if t <= model['d_start_exp']:
                ξ_U_t = 0
                ξ_P_t = 0
                ξ_N_t = 0
                ξ_R_t = 0

                r_U_t = 0
                r_P_t = 0
                r_N_t = 0
                r_R_t = 0

            elif t >= self.d_vaccine:
                ξ_U_t = model['ξ_U']
                ξ_P_t = model['ξ_P']
                ξ_N_t = model['ξ_N']
                ξ_R_t = 0.

                r_U_t = model['r_U']
                r_P_t = model['r_P']
                r_N_t = model['r_N']
                r_R_t = model['r_R']

            else:
                ξ_U_t = model['ξ_U']
                ξ_P_t = model['ξ_P']
                ξ_N_t = model['ξ_N']
                ξ_R_t = model['ξ_R']

                r_U_t = model['r_U']
                r_P_t = model['r_P']
                r_N_t = model['r_N']
                r_R_t = model['r_R']

            transition_matrix_t         = np.zeros((13,13))

            transition_matrix_t[0,1]    = ξ_U_t
            transition_matrix_t[0,2]    = model['τA']
            transition_matrix_t[0,4]    = self.λ*alphat

            transition_matrix_t[1,0]    = r_U_t
            transition_matrix_t[1,3]    = model['τA']
            transition_matrix_t[1,5]    = self.λQ*alphat

            transition_matrix_t[2,3]    = ξ_N_t
            transition_matrix_t[2,6]    = self.λ*alphat

            transition_matrix_t[3,2]    = r_N_t
            transition_matrix_t[3,7]    = self.λQ*alphat

            transition_matrix_t[4,5]    = ξ_U_t
            transition_matrix_t[4,6]    = model['τA']
            transition_matrix_t[4,8]    = self.δ

            transition_matrix_t[5,4]    = r_U_t
            transition_matrix_t[5,7]    = model['τA']
            transition_matrix_t[5,9]    = self.δ

            transition_matrix_t[6,7]    = ξ_P_t
            transition_matrix_t[6,8]    = self.δ

            transition_matrix_t[7,6]    = r_P_t
            transition_matrix_t[7,9]    = self.δ

            transition_matrix_t[8,9]    = ξ_P_t
            transition_matrix_t[8,10]   = self.ωR
            transition_matrix_t[8,12]   = self.ωD

            transition_matrix_t[9,8]    = r_P_t
            transition_matrix_t[9,11]   = self.ωR
            transition_matrix_t[9,12]   = self.ωD

            transition_matrix_t[10,4]    = self.γ * self.λ*alphat
            transition_matrix_t[10,11]   = ξ_R_t

            transition_matrix_t[11,5]    = self.γ * self.λ*alphat
            transition_matrix_t[11,10]   = r_R_t

            if t >= self.d_vaccine:
                transition_matrix_t[0,10] = .001
                transition_matrix_t[1,10] = .001
                transition_matrix_t[2,10] = .001
                transition_matrix_t[3,10] = .001

            transition_matrix_t += np.diag(1 - np.sum(transition_matrix_t, axis=1))

            assert np.min(transition_matrix_t) >= 0
            assert np.max(transition_matrix_t) <= 1

            M_t[:,t] = transition_matrix_t.T @ Mt

        Y_t                 = np.sum(M_t[[0,2,4,6,10]], axis=0) + \
                                self.A_rel * np.sum(M_t[[1,3,5,7,11]], axis=0)
        Reported_T_start    = self.pop * (model['τA'] + self.δ) * (M_t[4] + M_t[5])
        Reported_T_start[0] = 0
        Reported_T          = np.cumsum(Reported_T_start)

        Reported_D      = Reported_T[13::14]
        Notinfected_D   = np.sum(M_t[[0,1,2,3]], axis=0)[13::14]
        Unreported_D    = np.sum(M_t[[4,5]], axis=0)[13::14]
        Infected_D      = np.sum(M_t[[8,9]], axis=0)[13::14]
        Recovered_D     = np.sum(M_t[[10,11]], axis=0)[13::14]
        Dead_D          = M_t[12][13::14]
        Infected_T      = np.sum(M_t[4:10], axis=0)
        Y_D             = Y_t[13::14]
        
        # This is the function that we called in MATLAB
        return Reported_D, Notinfected_D, Unreported_D, Infected_D, \
                Recovered_D, Dead_D, Infected_T, Y_D, M_t

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
    def solve_model(self):
        # 1. Baseline
        Reported_D_base, Notinfected_D_base, Unreported_D_base, Infected_D_base, \
                Recovered_D_base, Dead_D_base, Infected_T_base, Y_D_base, M_t_base = \
                self.solve_case(self.baseline)
        # 2. Get Tstar
        Tstar = np.argwhere(Reported_D_base>100)[0][0]
        YearsPlot = 3
        Tplot = np.arange(Tstar, min(Tstar + YearsPlot * 365, self.T/self.Δ_time) + .5, 1)
        Xplot = np.arange(0, len(Tplot))
        self.Tstar = Tstar
        # 3. Common quarantine
        self.common_quarantine['d_start_exp'] = (Tstar+1) * self.Δ_time + \
                self.policy_offset * self.Δ_time
        
        Reported_D_com, Notinfected_D_com, Unreported_D_com, Infected_D_com, \
                Recovered_D_com, Dead_D_com, Infected_T_com, Y_D_com, M_t_com = \
                self.solve_case(self.common_quarantine)

        return Reported_D_base, Infected_D_base, Dead_D_base, Y_D_base, \
                Reported_D_com, Infected_D_com, Dead_D_com, Y_D_com
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def run_experiment(self, τ, Δ):
        # Runs the particular experiment

        τ_A_daily_target = τ

        r_U_daily_target	= 0
        r_N_daily_target	= 0
        r_P_daily_target	= 0
        r_R_daily_target	= self.r_high

        ξ_U_daily_target   = self.ξ_base
        ξ_P_daily_target   = self.ξ_base_high
        ξ_N_daily_target   = self.ξ_base*Δ
        ξ_R_daily_target   = 0

        self.test_and_quarantine = {
            'τA'            : (1+τ_A_daily_target)**(1./self.Δ_time)-1,
            'ξ_U'           : (1+ξ_U_daily_target)**(1./self.Δ_time)-1,
            'ξ_P'           : (1+ξ_P_daily_target)**(1./self.Δ_time)-1,
            'ξ_N'           : (1+ξ_N_daily_target)**(1./self.Δ_time)-1,
            'ξ_R'           : (1+ξ_R_daily_target)**(1./self.Δ_time)-1,
            'r_U'           : (1+r_U_daily_target)**(1./self.Δ_time)-1,
            'r_P'           : (1+r_P_daily_target)**(1./self.Δ_time)-1,
            'r_N'           : (1+r_N_daily_target)**(1./self.Δ_time)-1,
            'r_R'           : (1+r_R_daily_target)**(1./self.Δ_time)-1,
            'experiment'    : "baseline_vaccine_tag"
        }

        self.test_and_quarantine['d_start_exp'] = (self.Tstar+1) * self.Δ_time + \
                self.policy_offset * self.Δ_time

        Reported_D_test, Notinfected_D_test, Unreported_D_test, Infected_D_test, \
                Recovered_D_test, Dead_D_test, Infected_T_test, Y_D_test, M_t_test = \
                self.solve_case(self.test_and_quarantine)

        return Reported_D_test, Infected_D_test, Dead_D_test, Y_D_test

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is now outside the other class above
def generate_plots(Δ, τ, ξ_base, A_rel, d_vaccine, rel_ρ, δ_param, \
             ωR_param, π_D, R_0, rel_λ, initial_infect, slide_var):

    colors = ['blue', 'green', 'red']

    rmin = 0
    rmax = 0
    imin = 0
    imax = 0
    dmin = 0
    dmax = 0
    ymin = .5
    ymax = 0

    fig = make_subplots(2, 2, print_grid = False, \
                        subplot_titles=("Reported Cases", "Current Symptomatic Cases", "Deaths - Cumulative", "Current Output"))
    
    # Create corona model with parameters from the boxes
    model = corona_model(ξ_base, A_rel, d_vaccine, rel_ρ, δ_param, \
                 ωR_param, π_D, R_0, rel_λ, initial_infect)
    
    # Getting the baseline and common quarnatine
    Reported_D, Infected_D, Dead_D, Y_D, Reported_D_com, Infected_D_com, \
        Dead_D_com, Y_D_com = model.solve_model()

    rmin = min(rmin, min(np.min(Reported_D) * 1.2, \
                         np.min(Reported_D_com) * 1.2))
    rmax = max(rmax, max(np.max(Reported_D) * 1.2, \
                         np.max(Reported_D_com) * 1.2))
    imin = min(imin, min(np.min(Infected_D) * 1.2, \
                         np.min(Infected_D_com) * 1.2))
    imax = max(imax, max(np.max(Infected_D) * 1.2, \
                         np.max(Infected_D_com) * 1.2))
    dmin = min(dmin, min(np.min(Dead_D) * 1.2, \
                         np.min(Dead_D_com) * 1.2))
    dmax = max(dmax, max(np.max(Dead_D) * 1.2, \
                         np.max(Dead_D_com) * 1.2))
    ymin = min(ymin, min(np.min(Y_D) * 1.2, \
                         np.min(Y_D_com) * 1.2))
    ymax = max(ymax, max(np.max(Y_D) * 1.2, \
                         np.max(Y_D_com) * 1.2))

    fig.add_scatter(y = Reported_D, row = 1, col = 1, visible = True,
                    name = 'Baseline', line = dict(color = (colors[0]), width = 3))
    fig.add_scatter(y = Infected_D, row = 1, col = 2, visible = True,
                    name = 'Baseline', line = dict(color = (colors[0]), width = 3))
    fig.add_scatter(y = Dead_D, row = 2, col = 1, visible = True,
                    name = 'Baseline', line = dict(color = (colors[0]), width = 3))
    fig.add_scatter(y = Y_D, row = 2, col = 2, visible = True,
                    name = 'Baseline', line = dict(color = (colors[0]), width = 3))

    fig.add_scatter(y = Reported_D_com, row = 1, col = 1, visible = True,
                    name = 'Common Quarantine', line = dict(color = (colors[1]), width = 3))
    fig.add_scatter(y = Infected_D_com, row = 1, col = 2, visible = True,
                    name = 'Common Quarantine', line = dict(color = (colors[1]), width = 3))
    fig.add_scatter(y = Dead_D_com, row = 2, col = 1, visible = True,
                    name = 'Common Quarantine', line = dict(color = (colors[1]), width = 3))
    fig.add_scatter(y = Y_D_com, row = 2, col = 2, visible = True,
                    name = 'Common Quarantine', line = dict(color = (colors[1]), width = 3))
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if slide_var == 1: #Slide over τ
        prd            = product(τ, [Δ])
        slider_vars    = τ
        slider_varname = "τ"
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if slide_var == 2: #Slide over Δ
        prd            = product([τ], Δ)
        slider_vars    = Δ
        slider_varname = "Δ"
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Uses multi processing to solve this faster
    pool    = Pool(os.cpu_count())
    results = pool.starmap(model.run_experiment, prd)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for j in range(len(slider_vars)):

        rmin = min(rmin, np.min(results[j][0]) * 1.2)
        rmax = max(rmax, np.max(results[j][0]) * 1.2)
        imin = min(imin, np.min(results[j][1]) * 1.2)
        imax = max(imax, np.max(results[j][1]) * 1.2)
        dmin = min(dmin, np.min(results[j][2]) * 1.2)
        dmax = max(dmax, np.max(results[j][2]) * 1.2)
        ymin = min(ymin, np.min(results[j][3]) * 1.2)
        ymax = max(ymax, np.max(results[j][3]) * 1.2)

        fig.add_scatter(y = results[j][0], row = 1, col = 1, visible = j == 0,
                        name = 'Quarantine & Test', line = dict(color = (colors[2]), width = 3))
        fig.add_scatter(y = results[j][1], row = 1, col = 2, visible = j == 0,
                        name = 'Quarantine & Test', line = dict(color = (colors[2]), width = 3))
        fig.add_scatter(y = results[j][2], row = 2, col = 1, visible = j == 0,
                        name = 'Quarantine & Test', line = dict(color = (colors[2]), width = 3))
        fig.add_scatter(y = results[j][3], row = 2, col = 2, visible = j == 0,
                        name = 'Quarantine & Test', line = dict(color = (colors[2]), width = 3))

    steps = []
    for i in range(len(slider_vars)):
        step = dict(
            method = 'restyle',
            args = ['visible', ['legendonly'] * len(fig.data)],
            label = slider_varname + ' = '+'{}'.format(round(slider_vars[i], 3))
        )
        for j in range(8):
            step['args'][1][int(j)] = True
        for j in range(4):
            step['args'][1][8 + j + i * 4] = True
        steps.append(step)

    sliders = [dict(
        steps = steps
    )]

    fig.layout.sliders = sliders
    fig['layout'].update(height=800, width=1000,
                     title="Results", showlegend = False)

    fig['layout']['xaxis1'].update(range = [0, 60])
    fig['layout']['xaxis2'].update(range = [0, 600])
    fig['layout']['xaxis3'].update(range = [0, 600])
    fig['layout']['xaxis4'].update(range = [0, 600])

    fig['layout']['yaxis1'].update(title='Logarithm - Base 10', type='log', range = [rmin, np.log10(1_000_000)])
    fig['layout']['yaxis2'].update(title='Fraction of Initial Population', range=[imin, imax])
    fig['layout']['yaxis3'].update(title='Fraction of Initial Population', range = [dmin, dmax])
    fig['layout']['yaxis4'].update(title='Output', range = [ymin, 1.05])

    return fig

def generate_plots_2d(Δ, τ, ξ_base, A_rel, d_vaccine, rel_ρ, δ_param, \
             ωR_param, π_D, R_0, rel_λ, initial_infect):

    model = corona_model(ξ_base, A_rel, d_vaccine, rel_ρ, δ_param, \
                 ωR_param, π_D, R_0, rel_λ, initial_infect)

    Reported_D, Infected_D, Dead_D, Y_D, Reported_D_com, Infected_D_com, \
        Dead_D_com, Y_D_com = model.solve_model()

    prd = product(τ, Δ)

    pool = Pool(os.cpu_count())
    results = pool.starmap(model.run_experiment, prd)

    return Reported_D, Infected_D, Dead_D, Y_D, Reported_D_com, Infected_D_com, \
        Dead_D_com, Y_D_com, results, prd
