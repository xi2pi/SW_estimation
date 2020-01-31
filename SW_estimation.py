 
import numpy as np
#import pandas as pd
#from shutil import copyfile
 
from guidata.qt.QtGui import QMainWindow, QSplitter

from guidata.dataset.datatypes import DataSet
from guidata.dataset.dataitems import FileOpenItem
from guidata.configtools import get_icon
from guidata.qthelpers import create_action, add_actions, get_std_icon
 
import matplotlib.pyplot as plt
import sys
import pyqtgraph as PG
from PyQt4 import QtGui
from PyQt4 import QtCore
#from PyQt4.QtGui import * 
#from PyQt4.QtCore import *

from scipy.integrate import odeint
import scipy.optimize as optimization

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def find_emax(P_loop, V_loop, Vd):
    E_loop = P_loop/(V_loop-Vd)
    ind_emax = np.argmax(E_loop)
    return [max(E_loop), ind_emax]
    
def find_emin(P_loop, V_loop, Vd):
    E_loop = P_loop/(V_loop-Vd)
    ind_emin = np.argmin(E_loop)
    return [min(E_loop), ind_emin]

def compute_work(Emax, Emin, P, V):
    P_con = P * 133.322
    V_con = V * (10**-6)
    dV = V_con - np.roll(V_con, -1)
    print("Emin " + str(Emin[0] ))
    print("Emax " + str(Emax[0] ))
    print("Emin ind " + str(Emin[1] ))
    print("Emax ind " + str(Emax[1] ))
    if Emin[1] < 50:
        W_in = np.sum(np.multiply(P_con[Emin[1]:Emax[1]],  dV[Emin[1]:Emax[1]]))
        W_out = np.sum(np.multiply(P_con[Emax[1]:],  dV[Emax[1]:]))
    else:
        W_in = np.sum(np.multiply(P_con[:Emax[1]],  dV[:Emax[1]]))
        W_out = np.sum(np.multiply(P_con[Emax[1]:Emin[1]],  dV[Emax[1]:Emin[1]]))
        
    W_in = np.sum(np.multiply(P_con[:Emax[1]],  dV[:Emax[1]]))
    print("W in " + str(W_in))
    print("W out " + str(W_out))
    return(W_in-W_out)
    

# Model
class LPM_6elements(object):
    def __init__(self, times, y0=None):
        self._y0 = y0
        self._times = times
        
        self.e_lv_vector = 0
        self.q_lv_vector = 0
        self.q_sys_vector = 0
        self.v_lv_vector = 0
        
        self.p_lv_vector = 0
        self.p_rv_vector = 0


    def _simulate(self, parameters, init_states, times):
        Elv, Eao, Evc, Rc, Ri, Ro, SBV, T = [x for x in parameters]
        dVlv_0, dVao_0, dVvc_0 = [x for x in init_states]
        
        
        
        # defining the model ODE       
        def model_ode(y, t, p):
            Vlv, Vao, Vvc = y

            
            # model equations
            Pao = P_(Eao, Vao)
            Pvc = P_(Evc, Vvc) 
            
            # Plv = Elv*e_lv(t,T,Wlv)*Vlv 
            Plv = Elv*Esin(t, T)*Vlv
            
            Qc = Q_(Pao, Pvc, Rc)
            Qi = Q_valves_(Pvc, Plv, Ri)
            Qo = Q_valves_(Plv, Pao, Ro)
        
            
            # ODE
            dVlv = Qi - Qo  # dVlv/dt
            dVao = Qo - Qc  # dVao/dt
            dVvc = Qc - Qi  # dVvc/dt            
            return dVlv, dVao, dVvc
        
        # elastance of the left ventricle            
        def Esin(t, T):
            Tsys = 0.3*np.sqrt(T)   #s Samar (2005)
            Tir = 0.5*Tsys #s Datensatz IM
            heart_cycle = int(t/T)
        
            t = t - heart_cycle * T
            if ((Tsys + Tir < t) & (t<T)):
                return int(0)
            elif ((Tsys < t) & (t < Tsys + Tir)):
                return 0.5*(1+np.cos(np.pi*((t-Tsys)/Tir)))
            elif ((0 < t) & (t < Tsys)):
                return 0.5*(1-np.cos(np.pi*(t/Tsys)))
            else:
                return 0
        
        # volume flow through resistance
        def Q_(P1, P2, R):
            return (P1 - P2)/R 
          
        # volume flow through valve  
        def Q_valves_(P1, P2, R):
            if P1>P2:
                Q = (P1 - P2)/R 
            else:
                Q = 0
            return Q 
          
        # pressure  
        def P_(E, V):
             return E*V
                
        values = odeint(model_ode, [dVlv_0, dVao_0, dVvc_0], times, (parameters,))
        

       
        return values
        
    def simulate(self, x, y):
            return self._simulate(x, y, self._times)
            
def Esin(t, T):
    Tsys = 0.3*np.sqrt(T)   #s Samar (2005)
    Tir = 0.5*Tsys #s Datensatz IM
    heart_cycle = int(t/T)

    t = t - heart_cycle * T
    if ((Tsys + Tir < t) & (t<T)):
        return int(0)
    elif ((Tsys < t) & (t < Tsys + Tir)):
        return 0.5*(1+np.cos(np.pi*((t-Tsys)/Tir)))
        #return 0.5*(1-np.cos(((t-Tsys)/Tir)))
    elif ((0 < t) & (t < Tsys)):
        return 0.5*(1-np.cos(np.pi*(t/Tsys)))
        #return 0.5*(1-np.cos((t/Tsys)))
    else:
        return 0
             
#-----------------------------------
class SmoothGUI(DataSet):

    fname = FileOpenItem("Open file", ("txt", "csv"), "")
    
class TaskManager(QtCore.QObject):
    error_signal = QtCore.pyqtSignal(list)

    def __init__(self, parent=None):
        QtCore.QObject.__init__(self, parent)
        self.para_start = 0
        self.parameter = (0,0, 0, 0, 0, 0, 0, 0)
        self.error_vector = []
    
    def error_function(self, tuning_parameters, EDV, ESV, Plv, T, Eao, Evc, Rmt, Rao):
        Elv, Rsys, SBV = tuning_parameters
    
        Vlv_u = 0
    
        # defining initial conditions
        dVlv_0 = SBV/3
        dVao_0 = SBV/3
        dVvc_0 = SBV/3
        
        
        if all(i > 0 for i in tuning_parameters):
            # parameter 
            parameter = [Elv, Eao, Evc, Rsys, Rmt, Rao, SBV, T]
            
            init_states = [dVlv_0, dVao_0, dVvc_0]
            
            # find init
            n_times = 100
            FN_solver_times = np.linspace(0, 20, n_times)
            ode_model = LPM_6elements(FN_solver_times)
            volumes_model_init = ode_model.simulate(parameter, init_states)            
            
          
            # init states 
            #init_states = [dVao_0, dVvc_0, dVpa_0, dVpu_0, dVlv_0, dVrv_0]
            init_states = [volumes_model_init[:,0][-1], volumes_model_init[:,1][-1], volumes_model_init[:,2][-1]]
            
            n_times = 1000
            FN_solver_times = np.linspace(0, T*5, n_times)
            
            # model - simulation
            ode_model = LPM_6elements(FN_solver_times)
            volumes_model = ode_model.simulate(parameter, init_states)
            
            Vlv_vector = volumes_model[:,0] + Vlv_u
            
            ESV_modeled =  min(Vlv_vector[-200:])
            EDV_modeled =  max(Vlv_vector[-200:])
            
    
            Plv_modeled = [Elv*Esin(t_value, T)*v_value for t_value, v_value in zip(FN_solver_times, volumes_model[:,0])]
            
            
            error_ESV = (ESV - ESV_modeled)**2
            error_EDV = (EDV - EDV_modeled)**2
    

            error_Plv = (Plv - max(Plv_modeled[-200:]))**2
            
            error_total = error_ESV + error_EDV + error_Plv 
        else:
            error_total = np.inf
    
        print(error_total)
        
        self.error_vector.append(error_total)
        self.error_signal.emit(self.error_vector)
        return error_total
                
    

    def start_process(self):
        res_LPM =  optimization.minimize(self.error_function,
                                         self.para_start, 
                                         args=self.parameter,
                                         method='Nelder-Mead')
                                         

    


#-----------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowIcon(get_icon('python.png'))
        self.setWindowTitle("Stroke Work Estimation")
        
        pal=QtGui.QPalette()
        role = QtGui.QPalette.Background
        pal.setColor(role, QtGui.QColor(255, 255, 255))
        self.setPalette(pal)
       
        self.textEdit = QtGui.QLabel('None')

        self.loadButton = QtGui.QPushButton("Compute")
        self.loadButton.clicked.connect(self.on_click)
        self.buttonSave = QtGui.QPushButton('Clear Plot', self)
        self.buttonSave.clicked.connect(self.clearPlots)
        
        self.dropDownMenu = QtGui.QComboBox()
        self.dropDownMenu.addItems(["Vao", "Vvc", "Vpa", "Vpu", "Vlv", "Vrv", "all"])
        


        self.table1 = QtGui.QTableWidget()
        self.table2 = QtGui.QTableWidget()
         
        self.fileName = ''
        self.lastClicked = []
        self.number_plots = 0       

        
        self.pw1 = PG.PlotWidget(name='VTC')
        self.pw1.setLabel('left', 'Pressure', units='mmHg')
        self.pw1.setLabel('bottom', 'Volume', units='mL')
        self.pw1.setBackground((255,255,255))
        
        self.pw2 = PG.PlotWidget(name='VTC')
        self.pw2.setBackground((255,255,255))
        self.pw2.setLabel('left', 'Error', units='mmHg')
        self.pw2.setLabel('bottom', 'Time', units='s')
        
        self.fig1 = FigureCanvas(Figure(figsize=(5, 3), tight_layout=True))
        self._fig1 = self.fig1.figure.subplots()
#        self._fig1.set_xlim([0, 300])
#        self._fig1.set_ylim([0, 10**6])
        self._fig1.grid()
        
        self.fig2 = FigureCanvas(Figure(figsize=(5, 3), tight_layout=True))
        self._fig2 = self.fig2.figure.subplots()
        self._fig2.set_xlim([0, 300])
        self._fig2.set_ylim([-1, 10**6])
        self._fig2.set_yscale('symlog')
        self._fig2.grid()
        
        self.fig3 = FigureCanvas(Figure(figsize=(5, 3), tight_layout=True))
        self._fig3 = self.fig3.figure.subplots()
        self._fig3_2 = self._fig3.twinx()
        self._fig3_3 = self._fig3.twinx()
        
        self._fig3.set_xlim([0, 300])
        #self._fig3.set_ylim([0, ])
        self._fig3.grid()
           
        #horizontalLayout = QtGui.QHBoxLayout(self)
        splitter = QSplitter(QtCore.Qt.Vertical)
        splitter2 = QSplitter(QtCore.Qt.Vertical)
        splitterH = QSplitter(QtCore.Qt.Horizontal)
        
        #splitter.addWidget(self.dropDownMenu)
        splitter.addWidget(self.loadButton)
        splitter.addWidget(self.fig1)
        splitter.addWidget(self.fig2)
        splitter.addWidget(self.fig3)
        
        splitter2.addWidget(self.table1)
        splitter2.addWidget(self.table2)
        splitter2.addWidget(self.buttonSave)
        
        splitterH.addWidget(splitter)
        splitterH.addWidget(splitter2)
        
        self.table1.setRowCount(9)
        self.table1.setColumnCount(2)
        
        self.table2.setRowCount(11)
        self.table2.setColumnCount(2)       
        
        self.table1.setItem(0,0, QtGui.QTableWidgetItem("Eao"))
        self.table1.setItem(0,1, QtGui.QTableWidgetItem("1.0"))
        self.table1.setItem(1,0, QtGui.QTableWidgetItem("Evc"))
        self.table1.setItem(1,1, QtGui.QTableWidgetItem("1.0"))
        self.table1.setItem(2,0, QtGui.QTableWidgetItem("Rmt"))
        self.table1.setItem(2,1, QtGui.QTableWidgetItem("0.1"))
        self.table1.setItem(3,0, QtGui.QTableWidgetItem("Rao"))
        self.table1.setItem(3,1, QtGui.QTableWidgetItem("0.1"))   
        
        self.table1.setItem(5,0, QtGui.QTableWidgetItem("P max"))
        self.table1.setItem(5,1, QtGui.QTableWidgetItem("100"))        
        self.table1.setItem(6,0, QtGui.QTableWidgetItem("V max"))
        self.table1.setItem(6,1, QtGui.QTableWidgetItem("100"))  
        self.table1.setItem(7,0, QtGui.QTableWidgetItem("V min"))
        self.table1.setItem(7,1, QtGui.QTableWidgetItem("70"))
        self.table1.setItem(8,0, QtGui.QTableWidgetItem("T"))
        self.table1.setItem(8,1, QtGui.QTableWidgetItem("0.5"))           
        
        ####
        self.table2.setItem(0,0, QtGui.QTableWidgetItem("SW 1"))
        self.table2.setItem(0,1, QtGui.QTableWidgetItem("-"))
        self.table2.setItem(1,0, QtGui.QTableWidgetItem("SW 2"))
        self.table2.setItem(1,1, QtGui.QTableWidgetItem("-"))
        self.table2.setItem(2,0, QtGui.QTableWidgetItem("SW 3"))
        self.table2.setItem(2,1, QtGui.QTableWidgetItem("-"))
        
        self.table2.setItem(4,0, QtGui.QTableWidgetItem("Elv"))
        self.table2.setItem(4,1, QtGui.QTableWidgetItem("-"))
        self.table2.setItem(5,0, QtGui.QTableWidgetItem("Rsys"))
        self.table2.setItem(5,1, QtGui.QTableWidgetItem("-"))
        self.table2.setItem(6,0, QtGui.QTableWidgetItem("SBV"))
        self.table2.setItem(6,1, QtGui.QTableWidgetItem("-"))
        
        self.table2.setItem(8,0, QtGui.QTableWidgetItem("error"))
        self.table2.setItem(8,1, QtGui.QTableWidgetItem("-"))
        
        
        self.setCentralWidget(splitterH)
 
        self.setContentsMargins(10, 5, 10, 5)
        self.setGeometry(100, 100, 1000, 800)
        
        # File menu
        file_menu = self.menuBar().addMenu("File")
        quit_action = create_action(self, "Quit",
        shortcut="Ctrl+Q",
        icon=get_std_icon("DialogCloseButton"),
        tip="Quit application",
        triggered=self.close)
        add_actions(file_menu, (quit_action, ))
    


        
    def error_function(self, tuning_parameters, EDV, ESV, Plv, T, Eao, Evc, Rmt, Rao):
        Elv, Rsys, SBV = tuning_parameters
    
        Vlv_u = 0
    
        # defining initial conditions
        dVlv_0 = SBV/3
        dVao_0 = SBV/3
        dVvc_0 = SBV/3
        
        
        if all(i > 0 for i in tuning_parameters):
            # parameter 
            parameter = [Elv, Eao, Evc, Rsys, Rmt, Rao, SBV, T]
            
            init_states = [dVlv_0, dVao_0, dVvc_0]
            
            # find init
            n_times = 100
            FN_solver_times = np.linspace(0, 20, n_times)
            ode_model = LPM_6elements(FN_solver_times)
            volumes_model_init = ode_model.simulate(parameter, init_states)            
            
          
            # init states 
            #init_states = [dVao_0, dVvc_0, dVpa_0, dVpu_0, dVlv_0, dVrv_0]
            init_states = [volumes_model_init[:,0][-1], volumes_model_init[:,1][-1], volumes_model_init[:,2][-1]]
            
            n_times = 1000
            FN_solver_times = np.linspace(0, T*5, n_times)
            
            # model - simulation
            ode_model = LPM_6elements(FN_solver_times)
            volumes_model = ode_model.simulate(parameter, init_states)
            
            Vlv_vector = volumes_model[:,0] + Vlv_u
            
            ESV_modeled =  min(Vlv_vector[-200:])
            EDV_modeled =  max(Vlv_vector[-200:])
            
    
            Plv_modeled = [Elv*Esin(t_value, T)*v_value for t_value, v_value in zip(FN_solver_times, volumes_model[:,0])]
            
            
            error_ESV = (ESV - ESV_modeled)**2
            error_EDV = (EDV - EDV_modeled)**2
    

            error_Plv = (Plv - max(Plv_modeled[-200:]))**2
            
            error_total = error_ESV + error_EDV + error_Plv 
        else:
            error_total = np.inf
    
        print(error_total)
        self.error_vector.append(error_total)
        self.Elv_vector.append(Elv)
        self.Rsys_vector.append(Rsys)
        self.SBV_vector.append(SBV)
        
        self._fig1.clear()
        try:
            self._fig1.plot(Vlv_vector[-200:], Plv_modeled[-200:], "k")
        except:
            print("error")
            
        self._fig1.set_xlim([-5, 200])
        self._fig1.set_ylim([-10, 190])
        self._fig1.set_xlabel("Volume [ml]")
        self._fig1.set_ylabel("Pressure [mmHg]")
        self._fig1.hlines(Plv, -10, 200, linestyles='dashed')
        self._fig1.vlines(ESV, -10, 200, linestyles='dashed')
        self._fig1.vlines(EDV, -10, 200, linestyles='dashed')
        self._fig1.grid()
        self.fig1.draw()        
        
        self._fig2.clear()
        self._fig2.plot(self.error_vector, "k")
        self._fig2.set_xlim([0, 300])
        self._fig2.set_ylim([-1, 10**6])
        self._fig2.set_xlabel("Iteration [-]")
        self._fig2.set_ylabel(r"$f_{error}$ [-]")
        self._fig2.set_yscale('symlog')
        self._fig2.grid()
        self.fig2.draw()
        
        
        self._fig3.clear()
        self._fig3_2.clear()
        self._fig3_3.clear()
        self._fig3.plot(self.Elv_vector, "r")
        self._fig3_2.plot(self.Rsys_vector, "g")
        self._fig3_3.plot(self.SBV_vector, "b")        
        
        self._fig3.set_xlim([0, 300])
        self._fig3.set_ylim([0, 5])
        self._fig3_2.set_ylim([0, 5])
        self._fig3_3.set_ylim([0, 500])
        
        self._fig3.set_xlabel("Iteration [-]")
        self._fig3.set_ylabel(r"$E_{lv}$", color="r")
        self._fig3.tick_params(axis='y', labelcolor="r")
        self._fig3_2.set_ylabel(r"$R_{c}$", color="g")
        self._fig3_2.tick_params(axis='y', labelcolor="g")
        self._fig3_3.set_ylabel(r"$SBV$", color="b")
        self._fig3_3.tick_params(axis='y', labelcolor="b")
        self._fig3.grid()
        self._fig3_3.spines["right"].set_position(("axes", 1.2))
        make_patch_spines_invisible(self._fig3_3)
        # Second, show the right spine.
        self._fig3_3.spines["right"].set_visible(True)
        self.fig3.draw()        
        
        QtCore.QCoreApplication.processEvents()


        return error_total
        

    def clearPlots(self):
        self.pw1.clear()
        self.pw2.clear()
         
    def on_click(self):
        self.error_vector = []
        self.Elv_vector = []
        self.Rsys_vector = []
        self.SBV_vector = []
        
#        self.pw1.addLine(x=None, y=float(self.table1.item(5,1).text()) , pen=PG.mkPen('r', width=3))
#        self.pw1.addLine(x=float(self.table1.item(6,1).text()), y=None, pen=PG.mkPen('r', width=3))
#        self.pw1.addLine(x=float(self.table1.item(7,1).text()), y=None, pen=PG.mkPen('r', width=3))
        # parameters
        Eao = float(self.table1.item(0,1).text()) 
        Evc = float(self.table1.item(1,1).text())    

        Rmt = float(self.table1.item(2,1).text()) 
        Rao = float(self.table1.item(3,1).text())
        
        Psys = float(self.table1.item(5,1).text())
        EDV = float(self.table1.item(6,1).text())        
        ESV = float(self.table1.item(7,1).text())
        T = float(self.table1.item(8,1).text())
        
        para_start = [2, 0.8, 500]
        
#        process = QProcess(self)
#        process.finished.connect(self.onFinished)
#        process.startDetached(command, args)
        
        res_LPM =  optimization.minimize(self.error_function,
                                         para_start, 
                                         args=(EDV, ESV, Psys, T, Eao, Evc, Rmt, Rao),
                                         method='Nelder-Mead')
        
        
#        manager = TaskManager()
#        manager.para_start = para_start
#        manager.parameter = (EDV, ESV, Psys, T, Eao, Evc, Rmt, Rao)
#        manager.start_process()
        
#        manager.resultsChanged.connect(self.on_finished)
#        process = QtCore.QProcess(self)
#        process.readyReadStandardOutput.connect(self.plotError)
#        process.start(command, cmd_arg, QtCore.QIODevice.ReadOnly)


        Elv = res_LPM.x[0]
        Rsys =  res_LPM.x[1]
        SBV = res_LPM.x[2]
        
        Vlv_u = 0
        
        
        dVlv_0 = SBV/3
        dVao_0 = SBV/3
        dVvc_0 = SBV/3
        
        parameter = [Elv, Eao, Evc, Rsys, Rmt, Rao, SBV, T]
        
        init_states = [dVlv_0, dVao_0, dVvc_0]
        
        # find init
        n_times = 100
        FN_solver_times = np.linspace(0, 20, n_times)
        ode_model = LPM_6elements(FN_solver_times)
        volumes_model_init = ode_model.simulate(parameter, init_states)            
        
      
        # init states 
        init_states = [volumes_model_init[:,0][-1], volumes_model_init[:,1][-1], volumes_model_init[:,2][-1]]
        
        n_times = 1000
        FN_solver_times = np.linspace(0, 5*T, n_times)
        
        # model - simulation
        ode_model = LPM_6elements(FN_solver_times)
        volumes_model = ode_model.simulate(parameter, init_states)
        
        Vlv_vector = volumes_model[:,0] + Vlv_u
                
        Plv_modeled = [Elv*Esin(t_value, T)*v_value for t_value, v_value in zip(FN_solver_times, Vlv_vector[-200:])]
        
        Emax_cur = find_emax(np.array(Plv_modeled[-200:]), Vlv_vector[-200:], 0.0)
        Emin_cur = find_emin(np.array(Plv_modeled[-200:]), Vlv_vector[-200:], 0.0)
        
        ESV_modeled =  min(Vlv_vector[-200:])
        EDV_modeled =  max(Vlv_vector[-200:])        
        
        try:
            SW1 = compute_work(Emax_cur, Emin_cur, np.array(Plv_modeled[-200:]), Vlv_vector[-200:] )
        except:
            SW1 = 0
            
        col = ["b", "r", "k", "c", "m"]
        mypen = PG.mkPen(col[self.number_plots], width=3)
            
        self.pw1.plot(Vlv_vector[-200:], Plv_modeled[-200:], pen=mypen, lw=4, label="PV Loop")
        self.pw2.plot(np.linspace(0, T, 200), Plv_modeled[-200:], pen=mypen, lw=4, label="Pressure")
        
        SW2 = (max(Plv_modeled[-200:]) - min(Plv_modeled[-200:]))*(max(Vlv_vector[-200:]) - min(Vlv_vector[-200:])) * 133.322 * (10**-6)
         
        MAP = np.mean([Eao * Vao for Vao in volumes_model[:,1]])    
        SW3 = (EDV_modeled - ESV_modeled) * MAP * 133.322 * (10**-6)  
        
        self.table2.setItem(0,1, QtGui.QTableWidgetItem(str(round(SW1, 4))))
        self.table2.setItem(1,1, QtGui.QTableWidgetItem(str(round(SW2, 4))))
        self.table2.setItem(2,1, QtGui.QTableWidgetItem(str(round(SW3, 4))))
        
        self.table2.setItem(4,1, QtGui.QTableWidgetItem(str(round(Elv, 4))))
        self.table2.setItem(5,1, QtGui.QTableWidgetItem(str(round(Rsys, 4))))
        self.table2.setItem(6,1, QtGui.QTableWidgetItem(str(round(SBV, 4))))
        self.table2.setItem(8,1, QtGui.QTableWidgetItem(str(res_LPM.fun)))
        
        
        if self.number_plots < 4:
            self.number_plots += 1
        else:
            self.number_plots = 0
             
        print('Done')
        

        
if __name__ == '__main__':
    from guidata.qt.QtGui import QApplication
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())