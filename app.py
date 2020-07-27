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
    plt.ylim(-5,5)
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

plt.figure()
freqdomain = signal.fft()
sigfig = np.abs(freqdomain).plot()
plt.ylim(0,5)
st.pyplot(sigfig)


st.markdown("## Try a band-pass filter")


lowfreq = st.slider("Low frequency cut-off", 1, 49, 20)
highfreq = st.slider("High frequency cut-off", 50, 300, 100)
bp_data = signal.bandpass(lowfreq, highfreq)

plt.figure()
st.pyplot(bp_data.plot())

plt.figure()
freqdomain = bp_data.fft()
bpfig = np.abs(freqdomain).plot()
plt.ylim(0,5)
st.pyplot(bpfig)


st.markdown("## Try Whitening")

# white = signal.whiten()
white = sig1 / amp1 + sig2 / amp2 + sig3/amp3
st.pyplot(white.plot())

plt.figure()
freqdomain = white.fft()
whitefig = np.abs(freqdomain).plot()
plt.ylim(0,5)
st.pyplot(whitefig)

st.markdown("# Now try it with real data")

# -- Close all open figures
plt.close('all')

@st.cache   #-- Magic command to cache data
def load_gw(t0, detector):
    strain = TimeSeries.fetch_open_data(detector, t0-18, t0+18, cache=False)
    return strain


detector = 'H1'
t0 = 1126259462.4   #-- GW150914
strain = load_gw(t0, detector)
center = int(t0)
strain = strain.crop(center-16, center+16)

#-- Make a time series plot    

st.subheader('Raw data')
plt.figure()
fig1 = strain.plot()
plt.xlim(t0-0.2, t0+0.1)
st.pyplot(fig1)

# -- Plot psd
plt.figure()
psdfig = strain.psd().plot()
plt.xlim(10, 1800)
st.pyplot(psdfig)

# -- Try whitened and band-passed plot
# -- Whiten and bandpass data
st.subheader('Whitened and Bandbassed Data')
white_data = strain.whiten()
bp_data = white_data.bandpass(30, 400)
fig3 = bp_data.plot()
plt.xlim(t0-0.2, t0+0.1)
st.pyplot(fig3)


# -- PSD of whitened data
# -- Plot psd
plt.figure()
psdfig = bp_data.psd().plot()
plt.xlim(10, 1800)
st.pyplot(psdfig)


# -- Close all open figures
plt.close('all')
