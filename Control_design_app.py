from control.matlab import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import streamlit as st
import math

STATE_INI = 0
STATE_P_FILLED = 1
STATE_C_FILLED = 2

g = 9.81
l = 0.2
m = 0.5
c = 1.5e-2
J = 1.0e-2

if 'count' not in st.session_state:
    st.session_state.count = 0

# Functions
def show_st_step_wrap(s_funs_in, legs_in, target=1):
    try:
        st.pyplot(show_st_step(s_funs_in, legs_in, target))
    except:
        st.write("Can't show Bodo graph well")


def show_st_bode_wrap(s_funs_in, legs_in):
    try:
        st.pyplot(show_st_bode(s_funs_in, legs_in))
    except:
        st.write("Can't show Bodo graph well")

# @st.cache #Too slow than no cache
def show_st_step(s_funs_in, legs_in, target=1):
    s_funs = []
    legs = []

    t_max = 0
    y_max = 0
    y_min = 0
    thre = 0.05

    fig1 = plt.figure(figsize=(6, 6))
    ax = plt.subplot()

    if 'list' not in str(type(s_funs_in)):
        s_funs.append(s_funs_in)
        legs.append(legs_in)
    else:
        s_funs = s_funs_in
        legs = legs_in

    ax.set_xlabel('t')
    ax.set_ylabel('y')
    ax.grid(ls=":")

    for i, s_fun in enumerate(s_funs):

        # stepinfo() returns wrong value...
        #Info = stepinfo(s_fun,SettlingTimeThreshold=thre)
        # print(Info)
        #y,t = step(s_fun, np.arange(0,int(Info["SettlingTime"])*2,0.01))
        #ax.axvline(Info['PeakTime'], lw=1,ls='--', c='red')
        #ax.axhline(Info['Peak'], lw=1,ls='--', c='red')
        #ax.axvline(Info['SettlingTime'], lw=1,ls='--', c='green')
        #ax.plot(t,y, label=(f'{legs[i]} SetValue:{Info["SteadyStateValue"]:.2f} SetTime: {Info["SettlingTime"]:.2f}'))

        y, t = step(s_fun, np.arange(0, 10, 0.001))
        [maxId] = signal.argrelmax(y)

        # Calclate settling time instead of stepinfo()
        set_t = None
        for t_tmp, y_tmp in zip(reversed(t), reversed(y)):
            if not ((target - thre) < y_tmp < (target + thre)):
                set_t = t_tmp
                break

        # Check settling value instead of stepinfo()
        a, b = np.polyfit(t[maxId], y[maxId], 1)
        if a > 0:
            set_y = np.Inf
        else:
            set_y = y[-1]

        ax.axvline(set_t, lw=1, ls='--', c='green')
        ax.plot(t, y, label=(
            f'{legs[i]} SetValue:{set_y:.2f} SetTime: {set_t:.2f}'))
        ax.plot(t[maxId], y[maxId], 'bo')
        t_max = max(max(t), t_max)
        y_max = max(max(y), y_max)
        y_min = min(min(y), y_min)
    ax.axhline(y=target+thre, lw=1, ls='--', c='green')
    ax.axhline(y=target-thre, lw=1, ls='--', c='green')
    ax.set_xlim(max(-1, -max(t[maxId])*0.1), max(t[maxId])*2)
    #ax.set_xlim(-1, t_max)
    ax.set_ylim(y_min - y_min * 0.1, y_max + y_max * 0.1)
    ax.legend()
    return(fig1)

# @st.cache #Too slow than no cache
def show_st_bode(s_funs_in, legs_in):
    s_funs = []
    legs = []
    if 'list' not in str(type(s_funs_in)):
        s_funs.append(s_funs_in)
        legs.append(legs_in)
    else:
        s_funs = s_funs_in
        legs = legs_in

    tmp_margin = [margin(s_fun) for s_fun in s_funs]
    gm = [t[0] for t in tmp_margin]
    pm = [t[1] for t in tmp_margin]
    wcp = [t[2] for t in tmp_margin]
    wgc = [t[3] for t in tmp_margin]

    # check cross frequency and use for range of graph
    wgc_max = 1
    for wgc_tmp in wgc:
        if np.isnan(wgc_tmp) or np.isinf(wgc_tmp):
            pass
        else:
            wgc_max = max(wgc_max, wgc_tmp)

    spaceMax = max(3, math.ceil(0.5 + np.log10(wgc_max)))

    tmp_bode = [bode(s_fun, logspace(-2, spaceMax), plot=False)
                for s_fun in s_funs]
    gain = [t[0] for t in tmp_bode]
    phase = [t[1] for t in tmp_bode]
    w = [t[2] for t in tmp_bode]

    fig1 = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    #gain, phase, w = bode(s_funs, logspace(-2,3), plot=False)

    for i, s_fun in enumerate(s_funs):
        #gm,pm,wcp,wgc = margin(s_fun)
        ax1.semilogx(w[i], 20*np.log10(gain[i]),
                     label=(f'{legs[i]} ($\omega_c$ {wgc[i]:.2f}[r/s]'))
        ax1.scatter(wgc[i], 0)
        ax2.semilogx(w[i], phase[i]*180/np.pi,
                     label=(f'{legs[i]} (Margin {pm[i]:.2f}[deg])'))

    ax1.grid(which="both", ls=':')
    ax1.set_ylabel('Gain [dB]')
    ax1.axhline(y=0, lw=1, ls='--', c='green')
    ax1.legend()

    ax2.grid(which="both", ls=':')
    ax2.set_xlabel('$\omega$ [rad/s]')
    ax2.set_ylabel('Phase [deg]')
    ax2.axhline(y=-180, lw=1, ls='--', c='green')
    ax2.legend()
    # plt.tight_layout()
    return fig1


# Main #
st.title('Classical Control Design')
st.markdown('### Input Plant model (P) in transfer function representation')

s = tf('s')
P = s
P_text_num = st.text_input('Numerator(分子)', '1')
P_text_den = st.text_input('Denominator(分母)', 'J*s**2 + c*s + 10*g*0.5')

if st.button('Apply Plant model') or st.session_state.count >= STATE_P_FILLED:
    if st.session_state.count < STATE_P_FILLED:
        st.session_state.count = STATE_P_FILLED
    P_text = f'({P_text_num})/({P_text_den})'
    exec("P = " + P_text)

    try:
        P_mk = r'$${\LARGE P = ' + P._repr_latex_().replace('$$', '') + r'}$$'
    except:
        P_mk = P

    st.markdown(P_mk)
    st.markdown('#### Open loop Bode')
    show_st_bode_wrap(P, "Plant")

    st.markdown('### input Controller (C) in transfer function')
    C = s
    C_text_num = st.text_input('Numerator(分子)', '(10000) + (1)/s + 5 * s')
    C_text_den = st.text_input('Denominator(分母)', '1')

    if st.button('Apply Controller model') or st.session_state.count >= STATE_C_FILLED:
        if st.session_state.count < STATE_C_FILLED:
            st.session_state.count = STATE_C_FILLED

        C_text = f'({C_text_num})/({C_text_den})'
        exec("C = " + C_text)

        try:
            C_mk = r'$${\LARGE C = ' + \
                C._repr_latex_().replace('$$', '') + r'}$$'
        except:
            C_mk = C

        st.markdown(C_mk)
        st.markdown("### Overall Open loop (P * C)")
        Hpid = P * C
        Hpid_mk = r'$${\LARGE Hpid = ' + \
            Hpid._repr_latex_().replace('$$', '') + r'}$$'
        st.markdown(Hpid_mk)
        st.markdown('#### Open loop Bode (P * C)')
        show_st_bode_wrap([Hpid, P], ["PID Controller", "Plant"])
        st.markdown("### Overall Closed loop (P * C)")
        Hpid_fb = feedback(Hpid, 1)
        Hpid_fb_mk = r'$${\LARGE Hpid_fb = ' + \
            Hpid_fb._repr_latex_().replace('$$', '') + r'}$$'
        st.markdown(Hpid_fb_mk)
        st.markdown('#### Step response on closedloop')
        show_st_step_wrap(Hpid_fb, "PID Controller")
        show_st_bode_wrap(Hpid_fb, "PID Controller")
