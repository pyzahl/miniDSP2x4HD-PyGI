#!/usr/bin/env python3

###########################################################################
## The original VU meter is a passive electromechanical device, namely
## a 200 uA DC d'Arsonval movement ammeter fed from a full wave
## copper-oxide rectifier mounted within the meter case. The mass of the
## needle causes a relatively slow response, which in effect integrates
## the signal, with a rise time of 300 ms. 0 VU is equal to +4 [dBu], or
## 1.228 volts RMS across a 600 ohm load. 0 VU is often referred to as "0
## dB".[1] The meter was designed not to measure the signal, but to let
## users aim the signal level to a target level of 0 VU (sometimes
## labelled 100%), so it is not important that the device is non-linear
## and imprecise for low levels. In effect, the scale ranges from -20 VU
## to +3 VU, with -3 VU right in the middle. Purely electronic devices
## may emulate the response of the needle; they are VU-meters inasmuch as
## they respect the standard.

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib

import sys
import os   # use os because python IO is bugy

import cairo
import time
import fcntl
from threading import Timer

import struct
import array
import math
from numpy  import *
from pylab import *

from meterwidget import *
from scopewidget import *

# MiniDSP 2x4 HD
from board_2x4hd import Board2x4HD

wins = {}

updaterate = 200

mdsp = Board2x4HD("usbhid")

print (mdsp.getInLevels())
print (mdsp.getOutLevels())
print (mdsp._masterStatus())


def delete_event(win, event=None):
        win.hide()
        # don't destroy window -- just leave it hidden
        return True


class SignalScope():

    def __init__(self, _button):

        label = "Oscilloscope"
        name=label
        self.ss = 0
        self.block_length = 128
        self.restart_func = self.nop
        self.trigger = 0
        self.trigger_level = 0
        win = Gtk.Window()
        wins[name] = win
        v = Gtk.VBox(spacing=0)

        scope = Oscilloscope( Gtk.Label(label="XT"), v)
        scope.scope.set_wide (True)
        scope.set_subsample_factor(256)
        scope.show()
        scope.set_chinfo(["L", "R"])
        win.add(v)
        if not self.ss:
                self.block_length = 128
                #parent.mk3spm.set_recorder (self.block_length, self.trigger, self.trigger_level)

        table = Gtk.Table(n_rows=4, n_columns=2)
        #table.set_row_spacings(5)
        #table.set_col_spacings(5)
        v.pack_start (table, True, True, 0)
        table.show()

        tr=0
        lab = Gtk.Label(label="# Samples:")
        lab.show ()
        table.attach(lab, 2, 3, tr, tr+1)
        Samples = Gtk.Entry()
        Samples.set_text("%d"%self.block_length)
        table.attach(Samples, 2, 3, tr+1, tr+2)
        Samples.show()

        #                [lab1, menu1] = parent.make_signal_selector (DSP_SIGNAL_SCOPE_SIGNAL1_INPUT_ID, signal, "CH1: ", parent.global_vector_index)
        #                lab = GObject.new(Gtk.Label, label="CH1-scale:")
        #                lab1.show ()
        #                table.attach(lab1, 0, 1, tr, tr+1)
        Xscale = Gtk.Entry()
        Xscale.set_text("1.")
        table.attach(Xscale, 0, 1, tr+1, tr+2)
        Xscale.show()

        #                [signal,data,offset] = parent.mk3spm.query_module_signal_input(DSP_SIGNAL_SCOPE_SIGNAL2_INPUT_ID)
        #                [lab2, menu1] = parent.make_signal_selector (DSP_SIGNAL_SCOPE_SIGNAL2_INPUT_ID, signal, "CH2: ", parent.global_vector_index)
        ##                lab = GObject.new(Gtk.Label, label="CH2-scale:")
        #                lab2.show ()
        #                table.attach(lab2, 1, 2, tr, tr+1)
        Yscale = Gtk.Entry()
        Yscale.set_text("1.")
        table.attach(Yscale, 1, 2, tr+1, tr+2)
        Yscale.show()
        
        tr = tr+2
        labx0 = Gtk.Label(label="X-off:")
        labx0.show ()
        table.attach(labx0, 0, 1, tr, tr+1)
        Xoff = Gtk.Entry()
        Xoff.set_text("0")
        table.attach(Xoff, 0, 1, tr+1, tr+2)
        Xoff.show()
        laby0 = Gtk.Label(label="Y-off:")
        laby0.show ()
        table.attach(laby0, 1, 2, tr, tr+1)
        Yoff = Gtk.Entry()
        Yoff.set_text("0")
        table.attach(Yoff, 1, 2, tr+1, tr+2)
        Yoff.show()
        
        self.xdc = 0.
        self.ydc = 0.

        def readdata():
                self.block_length = 128

                delayline_0  = zeros(self.block_length)
                delayline_1  = zeros(self.block_length)
                
                s0 = np.array(delayline_0)/3276.7
                s1 = np.array(delayline_1)/3276.7
                return s0,s1
        
        def update_scope(set_data, xs, ys, x0, y0, num, x0l, y0l):
                #blck = parent.mk3spm.check_recorder_ready ()
                blck = -1
                if blck == -1:
                        n = self.block_length
                        #[xd, yd] = parent.mk3spm.read_recorder (n)
                        [xd, yd] = readdata ()
                        
                        #if not self.run:
                        #                save("mk3_recorder_xdata", xd)
                        #                save("mk3_recorder_ydata", yd)
                        #                scope.set_flash ("Saved: mk3_recorder_[xy]data")
                        
                        # auto subsample if big
                        nss = n
                        nraw = n
                        if n > 4096:
                                ss = int(n/2048)
                                end =  ss * int(len(xd)/ss)
                                nss = (int)(n/ss)
                                xd = mean(xd[:end].reshape(-1, ss), 1)
                                yd = mean(yd[:end].reshape(-1, ss), 1)
                                scope.set_info(["sub sampling: %d"%n + " by %d"%ss,
                                                "T = %g ms"%(n/22.)])
                        else:
                                scope.set_info(["T = %g ms"%(n/22.)])
                                
                        # change number samples?
                        try:
                                self.block_length = int(num())
                                if self.block_length < 64:
                                        self.block_length = 64
                                if self.block_length > 999999:
                                        print ("MAX BLOCK LEN is 999999")
                                        self.block_length = 1024
                        except ValueError:
                                self.block_length = 128

                        if self.block_length != n or self.ss:
                                self.run = False
                                self.run_button.set_label("RESTART")

                        ##if not self.ss:
                        ##parent.mk3spm.set_recorder (self.block_length, self.trigger, self.trigger_level)
                        #                                v = value * signal[SIG_D2U]
                        #                                maxv = (1<<31)*math.fabs(signal[SIG_D2U])
                        try:
                                xscale_div = float(xs())
                        except ValueError:
                                xscale_div = 1
                                
                        try:
                                yscale_div = float(ys())
                        except ValueError:
                                yscale_div = 1

                        n = nss
                        try:
                                self.xoff = float(x0())
                                for i in range (0, n, 8):
                                        self.xdc = 0.9 * self.xdc + 0.1 * xd[i] * 1. #Xsignal[SIG_D2U]
                                x0l("X-DC = %g"%self.xdc)
                        except ValueError:
                                for i in range (0, n, 8):
                                        self.xoff = 0.9 * self.xoff + 0.1 * xd[i] * 1. #Xsignal[SIG_D2U]
                                x0l("X-off: %g"%self.xoff)

                        try:
                                self.yoff = float(y0())
                                for i in range (0, n, 8):
                                        self.ydc = 0.9 * self.ydc + 0.1 * yd[i] * 1. #Ysignal[SIG_D2U]
                                y0l("Y-DC = %g"%self.ydc)
                        except ValueError:
                                for i in range (0, n, 8):
                                        self.yoff = 0.9 * self.yoff + 0.1 * yd[i] * 1. #Ysignal[SIG_D2U]
                                y0l("Y-off: %g"%self.yoff)

                        if math.fabs(xscale_div) > 0:
                                xd = -(xd * 1. - self.xoff) / xscale_div
                        else:
                                xd = xd * 0. # ZERO/OFF
                                        
                        self.trigger_level = self.xoff / 1

                        if math.fabs(yscale_div) > 0:
                                yd = -(yd * 1. - self.yoff) / yscale_div
                        else:
                                yd = yd * 0. # ZERO/OFF
                                
                        scope.set_scale ( { 
                                "CH1:": "%g"%xscale_div + " V",
                                "CH2:": "%g"%yscale_div + " V",
                                "Timebase:": "%g ms"%(nraw/22./20.) 
                        })
                                
                        scope.set_data (xd, yd)

                if self.mode > 1:
                        self.run_button.set_label("SINGLE")
                else:
                        scope.set_info(["waiting for trigger or data [%d]"%blck])
                        scope.queue_draw()

                return self.run

        def stop_update_scope (win, event=None):
                print ("STOP, hide.")
                win.hide()
                self.run = False
                return True

        def toggle_run_recorder (b):
                if self.run:
                        self.run = False
                        self.run_button.set_label("RUN")
                else:
                        self.restart_func ()
                        
                        #[Xsignal, Xdata, Offset] = parent.mk3spm.query_module_signal_input(DSP_SIGNAL_SCOPE_SIGNAL1_INPUT_ID)
                        #[Ysignal, Ydata, Offset] = parent.mk3spm.query_module_signal_input(DSP_SIGNAL_SCOPE_SIGNAL2_INPUT_ID)
                        #scope.set_chinfo([Xsignal[SIG_NAME], Ysignal[SIG_NAME]])
        
                        period = int(2.*self.block_length/22.)
                        if period < 200:
                                period = 200
                        
                        if self.mode < 2: 
                                self.run_button.set_label("STOP")
                                self.run = True
                        else:
                                self.run_button.set_label("ARMED")
                                #parent.mk3spm.set_recorder (self.block_length, self.trigger, self.trigger_level)
                                self.run = False

                        GLib.timeout_add (period, update_scope, scope, Xscale.get_text, Yscale.get_text, Xoff.get_text, Yoff.get_text, Samples.get_text, labx0.set_text, laby0.set_text)

        def toggle_trigger (b):
                if self.trigger == 0:
                        self.trigger = 1
                        self.trigger_button.set_label("TRIGGER POS")
                else:
                        if self.trigger > 0:
                                self.trigger = -1
                                self.trigger_button.set_label("TRIGGER NEG")
                        else:
                                self.trigger = 0
                                self.trigger_button.set_label("TRIGGER OFF")
                print (self.trigger, self.trigger_level)

        def toggle_mode (b):
                if self.mode == 0:
                        self.mode = 1
                        self.ss = 0
                        self.mode_button.set_label("T-Auto")
                else:
                        if self.mode == 1:
                                self.mode = 2
                                self.ss = 0
                                self.mode_button.set_label("T-Normal")
                        else:
                                if self.mode == 2:
                                        self.mode = 3
                                        self.ss = 1
                                        self.mode_button.set_label("T-Single")
                                else:
                                        self.mode = 0
                                        self.ss = 0
                                        self.mode_button.set_label("T-Free")

        self.run_button = Gtk.Button("STOP")
        self.run_button.connect("clicked", toggle_run_recorder)
        self.hb = Gtk.HBox()
        self.hb.pack_start (self.run_button, True, True, 0)
        self.mode_button = Gtk.Button("M: Free")
        self.mode=0 # 0 free, 1 auto, 2 nommal, 3 single
        self.mode_button.connect("clicked", toggle_mode)
        self.hb.pack_start (self.mode_button, True, True, 0)
        self.mode_button.show ()
        table.attach(self.hb, 2, 3, tr, tr+1)
        tr = tr+1

        self.trigger_button = Gtk.Button("TRIGGER-OFF")
        self.trigger_button.connect("clicked", toggle_trigger)
        self.trigger_button.show ()
        table.attach(self.trigger_button, 2, 3, tr, tr+1)
        
        self.run = False
        win.connect("delete_event", stop_update_scope)
        toggle_run_recorder (self.run_button)
        
        wins[name].show_all()

    def set_restart_callback (self, func):
        self.restart_func = func
                        
    def nop (self):
        return 0
                        


        
# -- n is decimation factor, is 1 (off) or 8 (on
def calc_iir_stop():
        ab = calc_iir_coef_stop ()
        return ab

def calc_iir_pass():
        ab = calc_iir_coef_pass ()
        return ab

def calc_iir(fg,n):
        ab = calc_iir_coef_1simple (fg/22000.*n)
        return ab

def calc_iir_biquad(fg,n):
        ab = calc_iir_coef_biquad_single (fg/22000.*n)
        return ab


##         Cookbook formulae for audio EQ biquad filter coefficients
##----------------------------------------------------------------------------
##           by Robert Bristow-Johnson  <rbj@audioimagination.com>


##All filter transfer functions were derived from analog prototypes (that
##are shown below for each EQ filter type) and had been digitized using the
##Bilinear Transform.  BLT frequency warping has been taken into account for
##both significant frequency relocation (this is the normal "prewarping" that
##is necessary when using the BLT) and for bandwidth readjustment (since the
##bandwidth is compressed when mapped from analog to digital using the BLT).

##First, given a biquad transfer function defined as:

##            b0 + b1*z^-1 + b2*z^-2
##    H(z) = ------------------------                                  (Eq 1)
##            a0 + a1*z^-1 + a2*z^-2

##This shows 6 coefficients instead of 5 so, depending on your architechture,
##you will likely normalize a0 to be 1 and perhaps also b0 to 1 (and collect
##that into an overall gain coefficient).  Then your transfer function would
##look like:

#### THIS IT IS HERE #####

##            (b0/a0) + (b1/a0)*z^-1 + (b2/a0)*z^-2
##    H(z) = ---------------------------------------                   (Eq 2)
##               1 + (a1/a0)*z^-1 + (a2/a0)*z^-2

##or

##                      1 + (b1/b0)*z^-1 + (b2/b0)*z^-2
##    H(z) = (b0/a0) * ---------------------------------               (Eq 3)
##                      1 + (a1/a0)*z^-1 + (a2/a0)*z^-2


##The most straight forward implementation would be the "Direct Form 1"
##(Eq 2):

##    y[n] = (b0/a0)*x[n] + (b1/a0)*x[n-1] + (b2/a0)*x[n-2]
##                        - (a1/a0)*y[n-1] - (a2/a0)*y[n-2]            (Eq 4)

##This is probably both the best and the easiest method to implement in the
##56K and other fixed-point or floating-point architechtures with a double
##wide accumulator.



##Begin with these user defined parameters:

##    Fs (the sampling frequency)

##    f0 ("wherever it's happenin', man."  Center Frequency or
##        Corner Frequency, or shelf midpoint frequency, depending
##        on which filter type.  The "significant frequency".)

##    dBgain (used only for peaking and shelving filters)

##    Q (the EE kind of definition, except for peakingEQ in which A*Q is
##        the classic EE Q.  That adjustment in definition was made so that
##        a boost of N dB followed by a cut of N dB for identical Q and
##        f0/Fs results in a precisely flat unity gain filter or "wire".)

##     _or_ BW, the bandwidth in octaves (between -3 dB frequencies for BPF
##        and notch or between midpoint (dBgain/2) gain frequencies for
##        peaking EQ)

##     _or_ S, a "shelf slope" parameter (for shelving EQ only).  When S = 1,
##        the shelf slope is as steep as it can be and remain monotonically
##        increasing or decreasing gain with frequency.  The shelf slope, in
##        dB/octave, remains proportional to S for all other values for a
##        fixed f0/Fs and dBgain.



##Then compute a few intermediate variables:
        
##    A  = sqrt( 10^(dBgain/20) )
##       =       10^(dBgain/40)     (for peaking and shelving EQ filters only)

##    w0 = 2*pi*f0/Fs

##    cos(w0)
##    sin(w0)

##    alpha = sin(w0)/(2*Q)                                       (case: Q)
##          = sin(w0)*sinh( ln(2)/2 * BW * w0/sin(w0) )           (case: BW)
##          = sin(w0)/2 * sqrt( (A + 1/A)*(1/S - 1) + 2 )         (case: S)

##        FYI: The relationship between bandwidth and Q is
##             1/Q = 2*sinh(ln(2)/2*BW*w0/sin(w0))     (digital filter w BLT)
##        or   1/Q = 2*sinh(ln(2)/2*BW)             (analog filter prototype)

##        The relationship between shelf slope and Q is
##             1/Q = sqrt((A + 1/A)*(1/S - 1) + 2)

##    2*sqrt(A)*alpha  =  sin(w0) * sqrt( (A^2 + 1)*(1/S - 1) + 2*A )
##        is a handy intermediate variable for shelving EQ filters.


##Finally, compute the coefficients for whichever filter type you want:
##   (The analog prototypes, H(s), are shown for each filter
##        type for normalized frequency.)


##LPF:        H(s) = 1 / (s^2 + s/Q + 1)

##            b0 =  (1 - cos(w0))/2
##            b1 =   1 - cos(w0)
##            b2 =  (1 - cos(w0))/2
##            a0 =   1 + alpha
##            a1 =  -2*cos(w0)
##            a2 =   1 - alpha



##HPF:        H(s) = s^2 / (s^2 + s/Q + 1)

##            b0 =  (1 + cos(w0))/2
##            b1 = -(1 + cos(w0))
##            b2 =  (1 + cos(w0))/2
##            a0 =   1 + alpha
##            a1 =  -2*cos(w0)
##            a2 =   1 - alpha



##BPF:        H(s) = s / (s^2 + s/Q + 1)  (constant skirt gain, peak gain = Q)

##            b0 =   sin(w0)/2  =   Q*alpha
##            b1 =   0
##            b2 =  -sin(w0)/2  =  -Q*alpha
##            a0 =   1 + alpha
##            a1 =  -2*cos(w0)
##            a2 =   1 - alpha


##BPF:        H(s) = (s/Q) / (s^2 + s/Q + 1)      (constant 0 dB peak gain)

##            b0 =   alpha
##            b1 =   0
##            b2 =  -alpha
##            a0 =   1 + alpha
##            a1 =  -2*cos(w0)
##            a2 =   1 - alpha



##notch:      H(s) = (s^2 + 1) / (s^2 + s/Q + 1)

##            b0 =   1
##            b1 =  -2*cos(w0)
##            b2 =   1
##            a0 =   1 + alpha
##            a1 =  -2*cos(w0)
##            a2 =   1 - alpha



##APF:        H(s) = (s^2 - s/Q + 1) / (s^2 + s/Q + 1)

##            b0 =   1 - alpha
##            b1 =  -2*cos(w0)
##            b2 =   1 + alpha
##            a0 =   1 + alpha
##            a1 =  -2*cos(w0)
##            a2 =   1 - alpha



##peakingEQ:  H(s) = (s^2 + s*(A/Q) + 1) / (s^2 + s/(A*Q) + 1)

##            b0 =   1 + alpha*A
##            b1 =  -2*cos(w0)
##            b2 =   1 - alpha*A
##            a0 =   1 + alpha/A
##            a1 =  -2*cos(w0)
##            a2 =   1 - alpha/A



##lowShelf: H(s) = A * (s^2 + (sqrt(A)/Q)*s + A)/(A*s^2 + (sqrt(A)/Q)*s + 1)

##            b0 =    A*( (A+1) - (A-1)*cos(w0) + 2*sqrt(A)*alpha )
##            b1 =  2*A*( (A-1) - (A+1)*cos(w0)                   )
##            b2 =    A*( (A+1) - (A-1)*cos(w0) - 2*sqrt(A)*alpha )
##            a0 =        (A+1) + (A-1)*cos(w0) + 2*sqrt(A)*alpha
##            a1 =   -2*( (A-1) + (A+1)*cos(w0)                   )
##            a2 =        (A+1) + (A-1)*cos(w0) - 2*sqrt(A)*alpha



##highShelf: H(s) = A * (A*s^2 + (sqrt(A)/Q)*s + 1)/(s^2 + (sqrt(A)/Q)*s + A)

##            b0 =    A*( (A+1) + (A-1)*cos(w0) + 2*sqrt(A)*alpha )
##            b1 = -2*A*( (A-1) + (A+1)*cos(w0)                   )
##            b2 =    A*( (A+1) + (A-1)*cos(w0) - 2*sqrt(A)*alpha )
##            a0 =        (A+1) - (A-1)*cos(w0) + 2*sqrt(A)*alpha
##            a1 =    2*( (A-1) - (A+1)*cos(w0)                   )
##            a2 =        (A+1) - (A-1)*cos(w0) - 2*sqrt(A)*alpha

def calc_iir_coef_biquad_single(r):
        w = 2.*math.pi*r
        c = math.cos (w)
        s = math.sin (w)
        BW = 1.
        Q = 1./(2*math.sinh (math.log (2)/2.*BW*w/s))
        alpha = s/(2.*Q)
        a0 = 1 + alpha
        
        a=zeros( (3*4,4) )
        b=zeros( (3*4,4) )

        b[1][0] = (1-c)/2.   # b0
        b[1][1] =  1-c       # b1
        b[1][2] = (1-c)/2.   # b2

        # a0 DSP := 1 normed #  a0 = 1+alpha
        a[0][1] = -(-2*c)       # -a1
        a[0][2] = -(1-alpha)    # -a2

        a = a/a0
        b = b/a0

        b[0][0] = 1.         # 1
        b[3][0] = 1.         # 1
        b[6][0] = 1.
        b[9][0] = 1.

        print ("BiQuad LP calculated:")
        return [a,b]        


# H(s) = poly(An,s) / poly(Bn,s)
#
#                                  (10^(DSa_pass)) Prod_m(cB2(m,n))  
# HChebyshev,n (n even) (s) = ---------------------------------------------
#                                Prod_m(s^2 + cB1(m,n) + cB2(m,n)))
#
#                                  sinh(cD(n)) Prod_m(cB2(m,n))  
# HChebyshev,n (n odd) (s) = ---------------------------------------------
#                            (s+sinh(D)) Prod_m(s^2 + cB1(m,n) + cB2(m,n)))

## http://www.eng.iastate.edu/ee424/labs/C54Docs/spra079.pdf
## higher order: factorize into 2nd order filter products: H4 = H2_1 x H2_2 x ...
# use scilab filter design tools to calculate coefficients:
# Hlp=iir(4,'lp','cheb1',[75/22000/2,0],[0.001,0])
# nf=factors(numer(Hlp))  ==> poly (b[3k][0,1,2])
# df=factors(denom(Hlp))  ==> poly (a[3k][0,2,2])

def calc_iir_coef_stop():
        a=zeros( (3*4,4) )
        b=zeros( (3*4,4) )
        print ("stop IIR calculated:")
        return [a,b]

def calc_iir_coef_pass():
        C = 0
        a=zeros( (3*4,4) )
        b=zeros( (3*4,4) )
        a[0][1] = C
        a[3][1] = 0.

        b[0][0] = 1.-C
        b[0][1] = 0.

        b[3][0] = 1.
        b[6][0] = 1.
        b[9][0] = 1.
        print ("pass IIR calculated:")
        return [a,b]

def calc_iir_coef_1simple(r):
        C=math.exp(-r*2.*math.pi) # near 1
        print ("C=%g"%C)
        a=zeros( (3*4,4) )
        b=zeros( (3*4,4) )
        a[0][1] = C
        a[3][1] = 0.

        b[0][0] = 1.-C
        b[0][1] = 0.

        b[3][0] = 1.
        b[6][0] = 1.
        b[9][0] = 1.
        print ("simple LP calculated:")
        return [a,b]

def norm_iir_coef(a,b):
        # pre-scale
        # norm/limit to elements 0..1, split as needed
        print ("original")
        print ("a=")
        print (a)
        print ("b=")
        print (b)
        for ki in range(0,3):
                sa=1.
                sb=1.
                for i in range(0,4):
                        k=ki*3
                        sa = sa + math.fabs(a[k][i])
                        sb = sb + math.fabs(b[k][i])

                print ("Sum a,b[%d] = :"%ki + "(%f , " %sa + "%f)" %sb)

        a=a/4
        b=b/4

        print ("Scaled, Limited/Normed, Q15")
        # Q15
        q=32767.
        a=a*q
        b=b*q
        print ("a=")
        print (a)
        print ("b=")
        print (b)
        return [a,b]

def norm_iir_coef_XXX(a,b):
        # pre-scale
        a=a/2
        b=b/2
        # norm/limit to elements 0..1, split as needed
        for i in range(0,4):
                for ki in range(0,3):
                        k=ki*3
                        for q in range(0,2):
                                if a[k+q][i] > 1.:
                                        a[k+q+1][i]=a[k+q][i]-1.
                                        a[k+q][i]=1.
                                if a[k+q][i] < -1.:
                                        a[k+q+1][i]=a[k+q][i]+1.
                                        a[k+q][i]=-1.
                                if b[k+q][i] > 1.:
                                        b[k+q+1][i]=b[k+q][i]-1.
                                        b[k+q][i]=1.
                                if b[k+q][i] < -1.:
                                        b[k+q+1][i]=b[k+q][i]+1.
                                        b[k+q][i]=-1.
        print ("Scaled, Limited/Normed, Q15")
        # Q15
        q=32767.
        a=a*q
        b=b*q
        print ("a=")
        print (a)
        print ("b=")
        print (b)
        return [a,b]

#set gains for simple lowpass
def set_iir_stop(adr):
        set_iir_gains (calc_iir_stop())
def set_iir_pass(adr):
        set_iir_gains (calc_iir_pass())
def set_iir_low50(adr):
        set_iir_gains (calc_iir(50.,1))
def set_iir_low60(adr):
        set_iir_gains (calc_iir(60.,1))
def set_iir_low75(adr):
        set_iir_gains (calc_iir(75.,1))
def set_iir_low100(adr):
        set_iir_gains (calc_iir(100.,1))

#set gains for cheb low 60, 75, ..
def set_iir_lowbq60(adr):
        set_iir_gains (calc_iir_biquad(60.,8))
def set_iir_lowbq75(adr):
        set_iir_gains (calc_iir_biquad(75.,8))
def set_iir_lowbq100(adr):
        set_iir_gains (calc_iir_biquad(100.,8))



# create Gain edjustments
def create_control(_button):
        dbgain_address=0
        name="In"
        if name not in wins:
                win = Gtk.Window(title="Mini DSP Control")
                                  
                wins[name] = win
                win.connect("delete_event", delete_event)
                
                grid = Gtk.Grid()
                win.add(grid)
                grid.show()

                ##  getMute() setMute(mute)
                ##  get/setVolume()
                ##  get/setInputSource()
                ##  get/setConfig()
                ##  setGain() for 0,1 same  or use _setInputGain(input=0, gain=0)

                print ("CONFIG=", mdsp.getConfig())
                print (mdsp.getInputSource())
                
                grid.attach (Gtk.Label(label="Source:"), 1, 1, 1, 1)

                rb_analog = Gtk.RadioButton.new_with_label_from_widget(None, label="Analog")
                grid.attach (rb_analog, 2,1, 2,1)
                rb_usb    = Gtk.RadioButton.new_with_label_from_widget(rb_analog,  label="USB")
                grid.attach (rb_usb, 4,1, 2,1)
                rb_tos    = Gtk.RadioButton.new_with_label_from_widget(rb_usb,  label="TOS-Link")
                grid.attach (rb_tos, 6,1, 4,1)

                if mdsp.getInputSource() == 'analog':
                        rb_analog.set_active(True)
                elif mdsp.getInputSource() == 'usb':
                        rb_usb.set_active(True)
                elif mdsp.getInputSource() == 'toslink':
                        rb_tos.set_active(True)
                else:
                        print ("Input Source Invalid: " + str(mdsp.getInputSource()))

                def on_input(widget, data = None):
                        if rb_analog.get_active():
                                mdsp.setInputSource("analog")
                        elif rb_usb.get_active():
                                mdsp.setInputSource("usb")
                        elif rb_tos.get_active():
                                mdsp.setInputSource("toslink")
                            
                rb_analog.connect("toggled", on_input)
                rb_usb.connect("toggled", on_input)
                rb_tos.connect("toggled", on_input)

                
                grid.attach (Gtk.Label(label="Config:"), 1, 2, 1, 1)

                rb_c1 = Gtk.RadioButton.new_with_label_from_widget(None, label="C1")
                grid.attach (rb_c1, 2,2, 2,1)
                rb_c2 = Gtk.RadioButton.new_with_label_from_widget(rb_c1,  label="C2")
                grid.attach (rb_c2, 4,2, 2,1)
                rb_c3 = Gtk.RadioButton.new_with_label_from_widget(rb_c2,  label="C3")
                grid.attach (rb_c3, 6,2, 2,1)
                rb_c4 = Gtk.RadioButton.new_with_label_from_widget(rb_c3,  label="C3")
                grid.attach (rb_c4, 8,2, 2,1)

                if mdsp.getConfig() == 1:
                        rb_c1.set_active(True)
                elif mdsp.getConfig() == 2:
                        rb_c2.set_active(True)
                elif mdsp.getConfig() == 3:
                        rb_c3.set_active(True)
                elif mdsp.getConfig() == 4:
                        rb_c4.set_active(True)
                else:
                        print ("CONFIG Invalid: " + str(mdsp.getConfig()))
                

                
                def on_config(widget, data = None):
                        if rb_c1.get_active():
                                mdsp.setConfig(1)
                        elif rb_c2.get_active():
                                mdsp.setConfig(2)
                        elif rb_c3.get_active():
                                mdsp.setConfig(3)
                        elif rb_c4.get_active():
                                mdsp.setConfig(4)

                rb_c1.connect("toggled", on_config)
                rb_c2.connect("toggled", on_config)
                rb_c3.connect("toggled", on_config)
                rb_c4.connect("toggled", on_config)
                
                grid.attach (Gtk.Label(label="Volume:"), 1, 5, 1, 1)

                mute = Gtk.CheckButton (label="Mute")
                grid.attach (mute, 12, 5, 1, 1)
                print (mdsp.getMute())
                mute.set_active(mdsp.getMute())
     
                def on_mute(widget, data = None):
                        if mute.get_active():
                                mdsp.setMute(True)
                        else:
                                mdsp.setMute(False)
                                
                mute.connect("toggled", on_mute)
                
                adj=Gtk.Adjustment(value=mdsp.getVolume(), lower=-127.5, upper=0, step_increment=1, page_increment=3, page_size=0)
                volumescale = Gtk.HScale( adjustment=adj)
                volumescale.set_size_request(150, 40)
                volumescale.set_digits(1)
                volumescale.set_draw_value(True)
                grid.attach (volumescale, 2, 5, 10, 1)
                def volume_update(aj):
                        # print("V: {:f}".format(aj.get_value()))
                        mdsp.setVolume (aj.get_value())
                adj.connect("value-changed", volume_update)

                grid.attach (Gtk.Label(label="Gain:"), 1, 6, 1, 1)
                adj=Gtk.Adjustment(value=0, lower=-127.5, upper=0, step_increment=1, page_increment=3, page_size=0)
                gainscale = Gtk.HScale( adjustment=adj)
                gainscale.set_size_request(150, 40)
                gainscale.set_digits(1)
                gainscale.set_draw_value(True)
                grid.attach (gainscale, 2, 6, 10, 1)
                def gain_update(aj):
                        # print("G: {:f}".format(aj.get_value()))
                        mdsp.setGain (aj.get_value())
                adj.connect("value-changed", volume_update)

                grid.show_all()
        wins[name].show()

def create_vu_meters(levels_callback, name, Id):
        if name not in wins:
                win = Gtk.Window(title=name)
                wins[name] = win
                win.connect("delete_event", delete_event)
                
                grid = Gtk.Grid()
                win.add(grid)
                grid.show()
                
                db = array(levels_callback())
                v=[]
                c=[]
                for i in range(0, db.size):
                        v.append (Gtk.VBox())
                        v[0].show()
                        # c.append (Instrument( Gtk.Label(label=Id), v[i], "VU", "{:s} {:d}".format(Id,i), u="dB", widget_scale=0.75))
                        c.append (Instrument( Gtk.Label(label=Id), v[i], "VU", Id[i], u="dB", widget_scale=0.75))
                        if i > 1:
                                grid.attach (v[i], 1, i-2, 1, 1)
                        else:
                                grid.attach (v[i], 0, i, 1, 1)
                
                def update_meter(vu, levels_callback=levels_callback, count=[0]):
                        db = array(levels_callback())
                        for i in range(0,db.size):
                                vu[i].set_reading_vu (db[i], db[i])
                        return True

                GLib.timeout_add (77, update_meter, c)

        wins[name].show()
        
def create_vu_meter_in(_button):
        create_vu_meters(mdsp.getInLevels, "VU Inputs", ["In 0", "In 1"])

def create_vu_meter_out(_button):
        create_vu_meters(mdsp.getOutLevels, "VU Outputs", ["Out 0","Out 1","Out 2","Out 3"])

def create_vu_meter_inout(_button):
        create_vu_meters(mdsp.getLevels, "VU In+Outputs", ["In 0", "In 1", "Out 0", "Out 1", "Out 2", "Out 3"])


def do_exit(button):
        Gtk.main_quit()

def destroy(*args):
        Gtk.main_quit()

        
def create_main_window():
        buttons = {
                "Control": create_control,
                "VU Meter IN": create_vu_meter_in,
                "VU Meter OUT": create_vu_meter_out,
                "VU Meter IN+Out": create_vu_meter_inout,
         }
        win = Gtk.Window(title="Mini DSP 2x4HD")
        win.set_name("main window")
        win.set_size_request(250, 250)
        win.connect("destroy", destroy)
        win.connect("delete_event", destroy)

        box1 = Gtk.VBox()
        win.add(box1)
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_border_width(10)
        box1.pack_start(scrolled_window, True, True, 0)
        box2 = Gtk.VBox()
        box2.set_border_width(5)
        scrolled_window.add(box2)
        k = buttons.keys()
        for i in k:
                button = Gtk.Button(label=i)

                if buttons[i]:
                        button.connect("clicked", buttons[i])
                else:
                        button.set_sensitive(False)
                box2.pack_start(button, True, True, 0)

        separator = Gtk.HSeparator()
        box1.pack_start(separator, False, True, 0)
        box2 = Gtk.VBox(spacing=10)
        box2.set_border_width(5)
        box1.pack_start(box2, False, True, 0)
        button = Gtk.Button(label="close")
        button.connect("clicked", do_exit)
        box2.pack_start(button, True, True, 0)
        win.show_all()


print (__name__)
if __name__ == "__main__":
    print (sys.argv, len(sys.argv))
    if len(sys.argv) > 1:
        create_main_window()
        Gtk.main()
        print ("Byby.")
    else:
        create_main_window()
        Gtk.main()
        print ("Byby.")



