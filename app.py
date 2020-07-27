import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import requests, os
from gwpy.timeseries import TimeSeries
from gwosc.locate import get_urls
from gwosc import datasets

# -- Default detector list
detectorlist = ['H1','L1', 'V1']

# Title the app
st.title('Signal Processing Tutorial')

st.markdown("## Make a signal with 3 sine waves")


def makesine(freq, amp):
    fs = 4096
    time = np.arange(0,0.3, 1.0/fs)
    y1 = amp*np.sin( 2*np.pi*freq*time )
    sig1 = TimeSeries(y1, dt=1.0/fs).taper()
    fig_sig1 = sig1.plot()
    plt.xlim(0,0.3)
    st.pyplot(fig_sig1)

    return(sig1)

plt.figure()
freq1 = st.slider("Frequency", 20, 200, 20)
amp1 = st.slider("Amplitude", 1.0, 10.0, 5.0)
sig1 = makesine(freq1, amp1)

plt.figure()
freq2 = st.slider("Frequency", 20, 200, 47)
amp2 = st.slider("Amplitude", 1.0, 10.0, 2.0)
sig2 = makesine(freq2, amp2)

plt.figure()
freq3 = st.slider("Frequency", 20, 200, 195)
amp3 = st.slider("Amplitude", 1.0, 10.0, 1.0)
sig3 = makesine(freq3, amp3)

st.markdown("## Add the 3 sine waves together")
plt.figure()
signal = sig1 + sig2 + sig3
figsum = signal.plot()
plt.xlim(0,0.3)
st.pyplot(figsum)


st.markdown("## Convert to the frequency domain")

freqdomain = signal.fft()
st.pyplot(np.abs(freqdomain).plot())


