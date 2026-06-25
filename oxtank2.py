#Imports
import time as time_module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as scp
import rocketcea as rcea
from CoolProp.CoolProp import PropsSI
from rocketcea.cea_obj import CEA_Obj
import CoolProp.CoolProp as CP
import sys
import inputs_io
from rocketcea.cea_obj import CEA_Obj, add_new_fuel, add_new_oxidizer, add_new_propellant

class Oxtank():
    def __init__(self,RealTime,Oxtanktemp,NitrousQuality,StartMass_Gas,StartMass_Liquid,StartMass_TotalOx,Timestep,MetalOxtanktemp):
        self.time=RealTime
        self.Den_Gas=None
        self.Den_Liquid=None
        self.Vapour_PressurePa=None
        self.VapourPressure_PSI=None
        self.Den_Total=None
        self.gammanitrous=None
        self.fluid_Cp=None
        self.fluid_Cv=None
        self.time2=RealTime
        self.Oxtanktemp = Oxtanktemp
        self.NitrousQuality = NitrousQuality
        self.SpecificHeat=None
        self.PressureDiff_PA=None
        self.realgaspressure_PA=None
        self.molesN20Vape=None
        self.PressureDiff_Psi= None
        self.MassFlowrateSpi=None
        self.Vent_mdot=0
        self.Dloss=None
        self.RealMassFlowRate=None
        self.MdotOxpertick= None
        self.Vapourmass= StartMass_Gas
        self.liquidmassold=StartMass_Liquid
        self.totaloxmass= StartMass_TotalOx
        self.liquidmassflowrate=None
        self.vapourmassflowrate=0
        self.liquidmassnew=StartMass_Liquid
        self.AmountVaped=0
        self.status="liquid"
        self.fizz_X=0
        self.denFizzHelp=1
        self.cheat=False
        self.vapecounter=-4
        self.tempprime1=None
        self.tempcheatprime1=0
        self.tempcheatprime2=0
        self.tempcheatprime3=0
        self.tempprime1=None
        self.tempprime2=None
        self.tempprime3=None
        self.vapedprime1=None
        self.vapedprime2=None
        self.vapedprime3=None
        self.vapecomp=1/Timestep
        self.zfactor_Gas=None
        #self.Zfactor_calced=None
        self.tempprint=None
        self.vapstarttemp=None
        self.New_vapedencalc=None
        self.printdenvape=None
        self.zprint=None
        self.printvape_pressure=None
        self.currentpressureprintrepeat=None
        self.fizzvolumeratio=1
        self.currentfizzflowrate_voluem=None
        self.templost=0
        self.oxtankmetaltemp=MetalOxtanktemp
        self.nitrousViscosity=None
        self.volumeflowrate=None
        self.velocityox=None
        self.reynoldsnumberOx=None
        self.thermalconductivity=None
        self.prandtlnumber=None
        self.nusseltnumber=None
        self.heattransfercoefficient=None
        self.deltatemp=None
        self.heattransferrate_pertick=None
        self.changeintemp_Nitrous=None
        self.changeintemp_Oxtankmetal=None
        self.deltaZbasedOf_pressure=None
        self.deltaZbasedOf_density=None
        self.dZ_dT_constant_density=None
        self.dZ_dT_constant_pressure=None
        self.n_for2phaseflow=None
        self.mdot_spc=None
        self.trymdot=None
        self.difference_mdot=None
        self.downstream_liquidentropy=None
        self.downstream_Vapourentropy=None
        self.realupstream_entropy=PropsSI('S', 'T', self.Oxtanktemp, 'Q',self.NitrousQuality, 'NitrousOxide')
        self.downstream_enthalpy_liquid=None
        self.downstream_enthalpy_vapour=None
        self.downstream_enthalpy= None
        self.downstream_enthalpy2=None
        self.realupstream_enthalpy=None
        self.delta_enthalpy=None
        self.delta_enthalpy2=None
        self.mdot_hem_try1=None
        self.mdot_hem_try2=None
        self.mdot_hem_try3=None
        self.x_forOrfice_thermogenic=None
        self.x_forOrfice_thermogenic2=None
        self.densitydownstream=None
        self.densitydownstream2=None
        self.mdotchocked=None
        self.velocityhem=None
        self.velocityspi=None
        self.slip_velocity=None
        self.downstreamvoidfraction=None
        self.mdotfalala_notvap=None
        self.mdotfalala_notvap2=None
        self.MassFlowrateSpireapeat=None
        self.mdotfalala_vap=None
        self.mdotDryer=None
        self.CD_HEM=None
        self.CD_SPI=None
        self.InjectorVisc=None
        self.Temp_Injector=Oxtanktemp
        self.Reynoldsnumber_HEM=None
        self.Reynoldsnumber_SPI=None
        self.Den_forCD=None
        
        
        
        

        #csv stuff
        self._init_attrs = list(self.__dict__.keys())
        self.outputcsv = pd.DataFrame(columns=self._init_attrs)
        self._output_rows = []

    def StuffNoPrint(self, OxtankVolume, Timestep, oxtankLstart,hydroD, isventopen):
        self.fizznumber=0
        self.start_Fizzmass=1
        self.vapedprime1=None
        self.vapedprime2=None
        self.vapedprime3=None
        self.oldvape=0
        self.oldvape1=0
        self.oldvape2=0
        self.cheat=False
        self.tempprime1=None
        self.tempprime2=None
        self.tempprime3=None
        self.oldtemp=0
        self.oldtemp1=0
        self.oldtemp2=0
        self.tempcheat=0
        self.randomnumber=0
        self.tempcheatprime1=0
        self.tempcheatprime2=0
        self.tempcheatprime3=0
        self.SaturatedGas_Enthaply=None
        self.SaturatedLiquid_Enthaply=None
        self.VapourEnthapy=None
        self.vtank= OxtankVolume
        self.vapecomnumber=0.11/Timestep
        self.Kconstant=2 #could be 1/cd^2
        self.strartvapnumber=0
        self.gammanitrous=1.3
        self.vapstarttemp=None
        self.vapstartmass=None
        self.vapstartpressure=None
        self.vapstartdesnity=None
        self.vapstartzfactor=None
        self.VaptempNew=None
        self.zfactor_guess=None
        self.zloop=0
        self.correctz=None
        self.startvolumeflowratr_fizz=None
        self.lmassold_what=False
        self.lmassnew_what=False
        self.lstartoxtank=oxtankLstart
        self.sutherland_nitrous=263
        self.refernce_temp=184.68
        self.refernce_viscosity=0.0003237
        self.newvapphase=True
        self.vappressure_duringphase=None
        self.fizzcancled=True
        self.injectorhydrolicdiameter=hydroD
        self.injectorLength=1.2*2.54/100
        self.LDratio=self.injectorLength/self.injectorhydrolicdiameter
        self.Vent_happen=isventopen
        

        



    def cheatcheack(self):
        if self.cheat==False:
            if self.time>1:
                self.vapecomp=abs(self.vapedprime1/self.vapedprime2)
            if self.vapecomp<self.vapecomnumber:
                self.cheat=True
    

    def CoolOxtankProp (self,RealTime):
        
        self.Den_Gas= PropsSI('D', 'T', self.Oxtanktemp, 'Q', 1, 'NitrousOxide') #kg/m^3
        self.Den_Liquid= PropsSI('D', 'T', self.Oxtanktemp, 'Q', 0, 'NitrousOxide') #kg/M^3
        if self.status=="Vapour":
                if self.newvapphase==False:
                    self.Vapour_PressurePa=self.realgaspressure_PA
                else:
                    self.Vapour_PressurePa=self.vappressure_duringphase
        
        else:
            self.Vapour_PressurePa=PropsSI('P', 'T', self.Oxtanktemp, 'Q', 1, 'NitrousOxide') #PA
        self.VapourPressure_PSI= self.Vapour_PressurePa/6894.76
        self.SaturatedGas_Enthaply=PropsSI('H','T',self.Oxtanktemp,'Q',1,'NitrousOxide') 
        self.SaturatedLiquid_Enthaply=PropsSI('H','T',self.Oxtanktemp,'Q',0,'NitrousOxide')
        self.Den_Total=PropsSI('D', 'T', self.Oxtanktemp, 'Q', self.NitrousQuality, 'NitrousOxide')
        if self.status=="Vapour" and self.newvapphase==True:
            self.Den_Total=self.vapden
            self.Den_Gas=self.vapden
        
        self.VapourEnthapy= self.SaturatedGas_Enthaply-self.SaturatedLiquid_Enthaply
        self.SpecificHeat= PropsSI('C', 'T', self.Oxtanktemp, 'Q', 0, 'NitrousOxide')
        self.zfactor_Gas=PropsSI('Z', 'T', self.Oxtanktemp, 'Q',1, 'NitrousOxide')
        if self.status=="Vapour" and self.newvapphase==True:
            self.zfactor_Gas=self.correctz
        self.time=RealTime
        self.time2=RealTime
        self.fluid_Cp=PropsSI('Cp0mass', 'T', self.Oxtanktemp, 'S', self.realupstream_entropy, 'NitrousOxide')
        #self.fluid_Cv=PropsSI('Cvmass', 'T', self.Oxtanktemp, 'S', self.realupstream_entropy, 'NitrousOxide')
        self.fluid_Cv=self.fluid_Cp-188.9
        self.gammanitrous=self.fluid_Cp/self.fluid_Cv
        #self.gammanitrous=1.3
    def Massflowrate (self, Timestep, Is_2phase_flow_injector_model, DischargeCo_SPI, InjectorArea, chamberpressure_PA):
        
        self.PressureDiff_PA=self.Vapour_PressurePa-chamberpressure_PA #this is from fuel grian
        self.PressureDiff_Psi= self.PressureDiff_PA/6894.76
        self.Dloss= self.Kconstant/(InjectorArea**2)
        self.MassFlowrateSpi=(DischargeCo_SPI*InjectorArea*np.sqrt(2*self.Den_Total*self.PressureDiff_PA))
        if self.time>Timestep and Is_2phase_flow_injector_model==True:
            if self.status!="Vapour":
                self.RealMassFlowRate=self.mdotfalala_notvap
            else:
                self.RealMassFlowRate=self.mdotfalala_vap
        else:
            self.RealMassFlowRate=self.MassFlowrateSpi
        if Is_2phase_flow_injector_model==False:
            self.RealMassFlowRate=self.mdotDryer
    
        self.MdotOxpertick= self.RealMassFlowRate*Timestep
        self.liquidmassflowrate= self.RealMassFlowRate*Timestep*(1-self.NitrousQuality)
        if self.status == "Liquid":
            

            self.vapourmassflowrate=0
        elif self.status == "fizz":
            self.vapourmassflowrate=self.RealMassFlowRate*self.NitrousQuality*Timestep
        elif self.time>Timestep:
            self.vapourmassflowrate=self.RealMassFlowRate*Timestep
        else:
            self.vapourmassflowrate=0
        self.timestep=Timestep
        
        
    def Massesofshit(self):
        self.totaloxmass-=(self.liquidmassflowrate+self.vapourmassflowrate)
        if self.lmassold_what==False:
            self.liquidmassold= self.liquidmassnew-self.liquidmassflowrate
        if self.lmassnew_what==False:
            self.liquidmassnew= (self.vtank-(self.totaloxmass/self.Den_Gas))/((1/self.Den_Liquid)-(1/ self.Den_Gas))
        if self.liquidmassnew<=0:
            self.lmassnew_what=True
        if self.liquidmassold<=0:
            self.lmassold_what=True
        if self.lmassnew_what==True:
            self.liquidmassnew=0
        if self.lmassold_what==True:
            self.liquidmassold=0
        


        if self.status=="Liquid":
            self.Vapourmass=self.Vapourmass-self.vapourmassflowrate+self.AmountVaped
        elif self.status=="fizz":
            self.Vapourmass=self.Vapourmass-self.vapourmassflowrate+self.AmountVaped
        else:
            self.Vapourmass-=self.vapourmassflowrate

        
    def FindVapour(self,ventcd,ventarea,ambientpressure_pa):
        if self.Vent_happen==True:
            self.Vent_mdot=ventcd*ventarea*np.sqrt(2*self.Den_Gas*(self.Vapour_PressurePa-ambientpressure_pa))
            self.totaloxmass-=self.Vent_mdot*self.timestep
            #self.Vapourmass-=self.Vent_mdot*self.timestep
        self.AmountVaped=self.liquidmassold-self.liquidmassnew-(self.Vent_mdot*self.timestep)
    def Tempofnitrous(self):
        

        self.deltaQ= self.AmountVaped*self.VapourEnthapy
        if self.liquidmassnew!=0:
            self.templost= self.deltaQ/(self.liquidmassnew*self.SpecificHeat)
        else:
            self.templost=0
        if self.status !="Vapour":
            if self.cheat== True  and self.randomnumber!=0:
                self.Oxtanktemp=self.tempcheat
            else:
                self.Oxtanktemp= self.Oxtanktemp-self.templost
        if self.status=="Vapour":
            self.Oxtanktemp=self.tempprint
         
        
    
    def Derivatives(self):
        if self.time>0.1:
            self.vapedprime1=self.AmountVaped-self.oldvape
            self.oldvape=self.AmountVaped
            self.vapedprime2= self.vapedprime1-self.oldvape1
            self.oldvape1=self.vapedprime1
            self.vapedprime3= self.vapedprime2-self.oldvape2
            self.oldvape2=self.vapedprime2
        else:
            self.vapedprime1=0
            self.vapedprime2=0
            self.vapedprime3=0
        if self.vapedprime2<0 and self.time>1:
            self.vapecounter+=1
       # if self.vapecounter>0 and RealTime>0.01:
            #self.cheat=True

            
        if self.time>0.1:
            self.tempprime1=self.Oxtanktemp-self.oldtemp
            self.oldtemp=self.Oxtanktemp
            self.tempprime2= self.tempprime1-self.oldtemp1
            self.oldtemp1=self.tempprime1
            self.tempprime3= self.tempprime2-self.oldtemp2
            self.oldtemp2=self.tempprime2
        if self.cheat== True  and self.randomnumber==0:
            self.tempcheat=self.Oxtanktemp
            self.randomnumber=1
            self.tempcheatprime3=self.tempprime3
            self.tempcheatprime2=self.tempprime2
            self.tempcheatprime1=self.tempprime1
        elif self.cheat == True:
            #self.tempcheatprime2+=self.tempcheatprime3
            self.tempcheatprime1+=self.tempcheatprime2
            self.tempcheat+=self.tempcheatprime1
    def Status(self,StartMass_Liquid,OxamountwhenfizzstartsConstant,Is_fizz_when_equal,Fizzstart2):
        self.fizzmass=StartMass_Liquid*OxamountwhenfizzstartsConstant
        
        if Is_fizz_when_equal==True  :
            self.fizzmass=self.Vapourmass*Fizzstart2
        if self.liquidmassold>self.fizzmass:
            self.status="Liquid"
        elif self.liquidmassold>0:
            self.status="fizz"
            if self.fizzcancled==True:
                self.status="Liquid"
        else:
            self.status="Vapour"
    def Fizz(self,FizzCurveConstant):
        if self.status=="fizz" and self.fizznumber==0:
            self.start_Fizzmass=self.liquidmassold
            self.startvolumeflowratr_fizz=self.RealMassFlowRate/self.Den_Total
            self.fizznumber=1
        if self.status == "fizz":
            self.fizz_X= 1-(self.liquidmassold/self.start_Fizzmass)
            self.denFizzHelp=(np.sqrt(abs(1-self.fizz_X**FizzCurveConstant)))#**FizzCurveConstant2 maybe fix
            self.currentfizzflowrate_voluem=self.RealMassFlowRate/self.Den_Total
            self.fizzvolumeratio=self.currentfizzflowrate_voluem/self.startvolumeflowratr_fizz

        
    def NitrousQuality_func(self):
        if self.status=="Liquid":
            self.NitrousQuality=0
        elif self.status=="fizz":
            self.NitrousQuality= (1-self.denFizzHelp)
            if self.fizzcancled==True:
                self.NitrousQuality=0
        else:
            self.NitrousQuality=1
    def intailvapour(self):
        if self.status=="Vapour" and self.strartvapnumber==0 :
            
            self.vapstarttemp=self.Oxtanktemp
            self.vapstartmass=self.Vapourmass
            self.vapstartpressure=self.Vapour_PressurePa
            self.vapstartdesnity=self.Den_Gas
            self.vapstartzfactor=self.zfactor_Gas
            self.strartvapnumber=1
            self.VaptempNew=self.vapstarttemp
            self.vape_it_pres_pa=self.vapstartpressure
            self.oldpressurevape=self.Vapour_PressurePa
            self.vapetempold=self.Oxtanktemp
            self.templossCo=1
            self.vape_it_pres_pa=self.vapstartpressure
            self.startentropyspec=PropsSI('S', 'D|gas',  self.vapstartmass/self.vtank, 'T', self.vapstarttemp , 'NitrousOxide')
    def Vapourphase_constant_temp(self):
        self.molesN20Vape=self.Vapourmass*1000/44.031 #mole weight of nitrous
        self.realgaspressure_PA=(self.zfactor_Gas*self.molesN20Vape*self.Oxtanktemp*8.314)/self.vtank
        self.tempprint=self.vapstarttemp
    def VapourPhase(self,thrust):
        self.molesN20Vape=self.Vapourmass*1000/44.031 #mole weight of nitrous
        self.realgaspressure_PA=(self.zfactor_Gas*self.molesN20Vape*self.Oxtanktemp*8.314)/self.vtank
        self.zlist=[]
        self.whilezcount=0
        if self.status=="Vapour" and thrust>0: #Nozzlemath thrustreal
            if self.newvapphase==False:
                    while self.zloop==0:
                    # self.vape_it_pres_pa=PropsSI('P', 'T', self.VaptempNew, 'Q', 1, 'NitrousOxide')
                    # self.zfactor_guess=self.vape_it_pres_pa/(188.9*self.VaptempNew*self.Den_Total)
                        self.zfactor_guess=PropsSI('Z', 'T', self.VaptempNew, 'Q',1, 'NitrousOxide')
                        self.zlist.append(self.zfactor_guess)
                        self.zsum=0
                        for x in self.zlist:
                            self.zsum=self.zsum + x
                        self.zquessavg= self.zsum/len(self.zlist)
                        try:
                            # Check for invalid values before calculation
                            if self.Vapourmass <= 0 or self.zquessavg <= 0 or self.vapstarttemp <= 0:
                                self.VaptempNew = self.vapetempold
                            else:
                                self.VaptempNew=((((self.vapstartmass*self.vapstartzfactor)/(self.Vapourmass*self.zquessavg))**((self.gammanitrous-1)/(-1)))*self.vapstarttemp)
                                # Check if result is valid (not NaN or Inf)
                                if np.isnan(self.VaptempNew) or np.isinf(self.VaptempNew):
                                    self.VaptempNew = self.vapetempold
                        except:
                            self.VaptempNew=self.vapetempold
                        self.vapetempold=self.VaptempNew
                        #self.deltavapetemp=(self.vapetempold-self.VaptempNew)
                        #self.VaptempNew=self.vapetempold-(self.deltavapetemp*0.5)
                        #self.VaptempNew=PropsSI('T', 'Z', self.zquessavg, 'Q',1, 'NitrousOxide')
                        if self.VaptempNew<183:
                            self.VaptempNew=183
                            self.zloop=1
                        if self.VaptempNew>309:
                            self.VaptempNew=309
                            self.zloop=1
                        self.New_vapedencalc=((self.vapstarttemp/self.VaptempNew)**(1/(self.gammanitrous-1)))*self.vapstartdesnity
                        #self.vape_it_pres_pa=((self.vapstarttemp/self.VaptempNew)**(self.gammanitrous/(self.gammanitrous-1)))*self.vapstartpressure
                        
                        
                        #self.correctz=self.vape_it_pres_pa/(188.9*self.VaptempNew*self.New_vapedencalc)
                        self.correctz=PropsSI('Z', 'T', self.VaptempNew, 'Q',1, 'NitrousOxide')
                        #self.vape_it_pres_pa= (self.correctz*188.9*self.VaptempNew*self.New_vapedencalc)
                        self.vape_it_pres_pa=(self.correctz*8.314*self.VaptempNew/self.vtank*self.molesN20Vape)
                        #self.vape_it_pres_pa=PropsSI('P', 'T', self.VaptempNew, 'D', self.New_vapedencalc, 'NitrousOxide')
                        
                        self.redo=abs((self.correctz/self.zfactor_guess)-1)
                        self.whilezcount+=1
                        

                        
                        if self.redo<0.000001 :
                            self.zloop=1
                        if self.whilezcount==20:
                            self.zloop=1
                            self.whilezcount=0
                    self.tempprint=self.VaptempNew
                    #self.deltavapetemp=(self.vapetempold-self.VaptempNew)
                    #self.tempprint=self.vapetempold-(self.deltavapetemp*self.templossCo)
                    #self.vapetempold=self.tempprint
                    

                    

                    
                    self.zloop=0
                    #print(len(self.zlist))

                    #print(self.correctz)
                # print(self.VaptempNew)
                    self.zprint=PropsSI('Z', 'T', self.tempprint, 'Q',1, 'NitrousOxide')
                    self.zfactor_Gas=self.zprint
                    self.vape_it_pres_pa=(self.zprint*8.314*self.tempprint/self.vtank*self.molesN20Vape)

                    self.zlist=[]
                    self.New_vapeden=PropsSI('D', 'T', self.tempprint, 'Q', 1, 'NitrousOxide')
                    
                    self.printdenvape=self.New_vapeden
                    self.printvape_pressure=self.vape_it_pres_pa
                    self.currentpressureprintrepeat=self.Vapour_PressurePa
                    
                    
                    #self.New_vapedencalc=((self.vapstarttemp/self.VaptempNew)**(1/(self.gammanitrous-1)))*self.vapstartdesnity
                    #self.NewZ=self.Vapour_PressurePa/(188.9*self.VaptempNew*self.Den_Gas)
            else:
                self.vapden=self.totaloxmass/self.vtank
                self.tempprint=PropsSI('T', 'D|gas',  self.vapden, 'S', self.startentropyspec, 'NitrousOxide')
                self.vappressure_duringphase=PropsSI('P', 'D|gas',  self.vapden, 'S', self.startentropyspec, 'NitrousOxide')
                self.correctz=PropsSI('Z', 'D|gas',  self.vapden, 'S', self.startentropyspec, 'NitrousOxide')
        

    def heattransferoxtank(self,oxtankID,oxtanksurfaceareainput,oxtankmass,oxtankspecheat,Timestep):
        if self.status== "Vapour":
            self.nitrousViscosity=self.refernce_viscosity*((self.Oxtanktemp/self.refernce_temp)**1.5)*((self.refernce_temp+self.sutherland_nitrous)/(self.Oxtanktemp+self.sutherland_nitrous))
            self.volumeflowrate=self.RealMassFlowRate/self.Den_Total
            self.velocityox= self.volumeflowrate/(3.14*((oxtankID/2)**2))
            self.reynoldsnumberOx=(self.Den_Total*self.velocityox*oxtankID)/self.nitrousViscosity
            self.thermalconductivity=(0.08*self.Oxtanktemp-6.6 )/100 #fix for later TODO: Look at this formula and fix
            self.prandtlnumber=(self.SpecificHeat*self.nitrousViscosity)/self.thermalconductivity
            self.nusseltnumber=0.023*(self.reynoldsnumberOx**0.8)*(self.prandtlnumber**0.35)#idk last coefficent 
            self.heattransfercoefficient=(self.nusseltnumber*self.thermalconductivity)/oxtankID
            self.deltatemp=self.oxtankmetaltemp-self.Oxtanktemp
            self.heattransferrate_pertick=self.heattransfercoefficient*oxtanksurfaceareainput*self.deltatemp*Timestep
            self.changeintemp_Nitrous=self.heattransferrate_pertick/(self.SpecificHeat*self.totaloxmass)
            self.changeintemp_Oxtankmetal=self.heattransferrate_pertick/(oxtankmass*oxtankspecheat)
    def vapourphase_part_2(self):
        if self.status=="Vapour":
            self.wow="wow"
    
    def twophaseflow_genstuff(self,chamberpressure_PA,DischargeCo_SPI,InjectorArea,DischargeCo_HEM):
        if True:
            
            dT = 0.0001
            t = self.Oxtanktemp
            p = self.Vapour_PressurePa
            rho = self.Den_Gas
            
            
            # Density-based finite difference: (∂Z/∂T)_ρ
            try:
                Z1_rho = PropsSI('Z', 'D', rho, 'T', t - dT, 'NitrousOxide')
                Z2_rho = PropsSI('Z', 'D', rho, 'T', t + dT, 'NitrousOxide')
                self.deltaZbasedOf_density = (Z2_rho - Z1_rho) / (2 * dT)
            except:
                # If finite difference fails (near saturation), use single-sided or reduced dT
                try:
                    Z1_rho = PropsSI('Z', 'D', rho, 'T', t - dT*0.5, 'NitrousOxide')
                    Z2_rho = PropsSI('Z', 'D', rho, 'T', t, 'NitrousOxide')
                    self.deltaZbasedOf_density = (Z2_rho - Z1_rho) / (dT*0.5)
                except:
                    self.deltaZbasedOf_density = self.olddetaZbasedOf_density if hasattr(self, 'olddetaZbasedOf_density') else 0.0
            
            # Pressure-based finite difference: (∂Z/∂T)_p
            try:
                pdiiff=PropsSI('P', 'T', t , 'Q', self.NitrousQuality, 'NitrousOxide') - PropsSI('P', 'T', t - dT, 'Q', self.NitrousQuality, 'NitrousOxide')
                S1 = PropsSI('Q', 'P', p, 'T', t - dT, 'NitrousOxide')
                S2 = PropsSI('Q', 'P', p, 'T', t + dT, 'NitrousOxide')
                Z1 = PropsSI('Z', 'P', p, 'T', t + dT, 'NitrousOxide')
                Z2 = PropsSI('Z', 'P', p, 'T', t + dT*2, 'NitrousOxide')
                self.deltaZbasedOf_pressure = (Z2 - Z1) / (1 * dT)
                #print(Z1,Z2)
            except:
                # If finite difference fails (near saturation), use single-sided or reduced dT
                try:
                    Z1 = PropsSI('Z', 'P', p, 'T', t - dT*0.5, 'NitrousOxide')
                    Z2 = PropsSI('Z', 'P', p, 'T', t, 'NitrousOxide')
                    
                    self.deltaZbasedOf_pressure = (Z2 - Z1) / (dT*0.5)
                except:
                    self.deltaZbasedOf_pressure = self.olddetaZbasedOf_pressure if hasattr(self, 'olddetaZbasedOf_pressure') else 0.0
               # Thermodynamic derivatives
            # (∂Z/∂T)_p = (p/R*) * [1/T * (∂v/∂T)_p - v/T²] at constant pressure using vapor path
            try:
                R_specific = 8.314462618 / CP.PropsSI('M', 'NitrousOxide')
                dT_tiny = 0.0001
                # Use vapor (Q=1) to avoid saturation surface issues at constant pressure
                v = 1 / CP.PropsSI('Dmass', 'T', t+dT, 'P', p, 'NitrousOxide')
                
                v_plus = 1 / CP.PropsSI('Dmass', 'T', t + 2*dT, 'P', p, 'NitrousOxide')
                dvdT_p = (v_plus - v) / dT
                #dvdT_p2=R_specific / p
               # print(f"v: {v}, v_plus: {v_plus}, dvdT_p: {dvdT_p}")
                
                self.dZ_dT_constant_pressure = (p / R_specific) * (1/t * dvdT_p - v / (t**2))
            except Exception as e:
                # Fallback: set to zero if calculation fails
                self.dZ_dT_constant_pressure = 0.0
                    
            
            # (∂Z/∂T)_ρ = 1/(ρR*) * [1/T * (∂P/∂T)_ρ - P/T²] using CoolProp partials
            try:
                R_specific = 8.314462618 / CP.PropsSI('M', 'NitrousOxide')
                P_calc = CP.PropsSI('P', 'T', t, 'Dmass', rho, 'NitrousOxide')
                dPdT_rho = CP.PropsSI('d(P)/d(T)|D', 'T', t, 'Dmass', rho, 'NitrousOxide')
                self.dZ_dT_constant_density = 1 / (rho * R_specific) * (1/t * dPdT_rho - P_calc / (t**2))
            except Exception as e:
                print(f"Error in dZ_dT_constant_density: {e}")
                self.dZ_dT_constant_density = 0.0
            
            toppart=self.zfactor_Gas+self.Oxtanktemp*self.deltaZbasedOf_density
            self.olddetaZbasedOf_pressure=self.deltaZbasedOf_pressure
            self.olddetaZbasedOf_density=self.deltaZbasedOf_density
            bottompart=self.zfactor_Gas+self.Oxtanktemp*self.deltaZbasedOf_pressure
            if self.status!="Vapour" :
                self.n_for2phaseflow= self.gammanitrous*(toppart/bottompart)
            
                
            a=self.n_for2phaseflow/(self.n_for2phaseflow-1)
            b=(self.n_for2phaseflow+1)/self.n_for2phaseflow
            c=((chamberpressure_PA/self.Vapour_PressurePa)**(2/self.n_for2phaseflow))-((chamberpressure_PA/self.Vapour_PressurePa)**(b))
            self.mdot_spc=DischargeCo_SPI*InjectorArea*np.sqrt(2*self.Den_Total*self.Vapour_PressurePa*a*c)
            self.difference_mdot=self.mdot_spc - self.MassFlowrateSpi
            self.downstream_liquidentropy=PropsSI('S', 'P', chamberpressure_PA, 'Q',0, 'NitrousOxide')
            self.downstream_Vapourentropy=PropsSI('S', 'P', chamberpressure_PA, 'Q',1, 'NitrousOxide')
            self.realupstream_entropy=PropsSI('S', 'T', self.Oxtanktemp, 'Q',self.NitrousQuality, 'NitrousOxide')
            self.x_forOrfice_thermogenic=(self.realupstream_entropy - self.downstream_liquidentropy)/(self.downstream_Vapourentropy - self.downstream_liquidentropy)
            #self.x_forOrfice_thermogenic2=((self.downstream_Vapourentropy - self.realupstream_entropy)/(self.realupstream_entropy - self.downstream_liquidentropy))
            if self.x_forOrfice_thermogenic<0:
                self.x_forOrfice_thermogenic=0
            if self.x_forOrfice_thermogenic>1:
                self.x_forOrfice_thermogenic=1
            
            
            self.downstream_enthalpy_liquid=PropsSI('H', 'P', chamberpressure_PA, 'Q',0, 'NitrousOxide')
            self.downstream_enthalpy_vapour=PropsSI('H', 'P', chamberpressure_PA, 'Q',1, 'NitrousOxide')
            self.realupstream_enthalpy=PropsSI('H', 'P', self.Vapour_PressurePa, 'Q',self.NitrousQuality, 'NitrousOxide')
            self.downstream_enthalpy= self.x_forOrfice_thermogenic*self.downstream_enthalpy_vapour+(1-self.x_forOrfice_thermogenic)*self.downstream_enthalpy_liquid
            if self.status=="Vapour":
                self.realupstream_entropy=self.startentropyspec
                self.x_forOrfice_thermogenic=PropsSI('Q', 'S', self.startentropyspec, 'P', chamberpressure_PA, 'NitrousOxide')
                if self.x_forOrfice_thermogenic<=0:
                    print(f"Warning: Calculated vapor quality is negative ({self.x_forOrfice_thermogenic}). Setting to 0.")
                    self.x_forOrfice_thermogenic=1
                if self.x_forOrfice_thermogenic>1:
                    print(f"Warning: Calculated vapor quality is greater than 1 ({self.x_forOrfice_thermogenic}). Setting to 1.")
                    self.x_forOrfice_thermogenic=1
                self.downstream_enthalpy= PropsSI('H', 'P', chamberpressure_PA, 'S',self.startentropyspec, 'NitrousOxide')
                self.realupstream_enthalpy=PropsSI('H', 'P', self.Vapour_PressurePa, 'S',self.startentropyspec, 'NitrousOxide')
                
           # self.downstream_enthalpy2=self.x_forOrfice_thermogenic2*self.downstream_enthalpy_liquid+(1-self.x_forOrfice_thermogenic2)*self.downstream_enthalpy_vapour
            self.delta_enthalpy=self.realupstream_enthalpy - self.downstream_enthalpy
            #self.delta_enthalpy2=self.realupstream_enthalpy - self.downstream_enthalpy2
            self.velocityhem=np.sqrt(2*self.delta_enthalpy)
            self.velocityspi=np.sqrt(2*self.PressureDiff_PA/self.Den_Total)
           # self.mdot_hem_try1=self.Den_Total*InjectorArea*DischargeCo_SPI*np.sqrt(2*self.delta_enthalpy)
            self.densitydownstream=PropsSI('D', 'Q', self.x_forOrfice_thermogenic, 'P', chamberpressure_PA, 'NitrousOxide')
            self.densitydownstream_vap=PropsSI('D', 'P', chamberpressure_PA, 'Q',1, 'NitrousOxide')
            self.densitydownstream_liquid=PropsSI('D', 'P', chamberpressure_PA, 'Q',0, 'NitrousOxide')
            self.densitydownstream2=self.x_forOrfice_thermogenic*self.densitydownstream_vap+(1-self.x_forOrfice_thermogenic)*self.densitydownstream_liquid
            self.mdot_hem_try2=self.densitydownstream2*InjectorArea*DischargeCo_HEM*np.sqrt(2*self.delta_enthalpy)
            #self.mdot_hem_try3=self.densitydownstream2*InjectorArea*DischargeCo_HEM*np.sqrt(2*self.delta_enthalpy2)
            self.mdotchocked=DischargeCo_SPI*InjectorArea*np.sqrt(self.Den_Total*self.Vapour_PressurePa*self.n_for2phaseflow*((2/(self.n_for2phaseflow+1))**((self.n_for2phaseflow+1)/(self.n_for2phaseflow-1))))
            self.slip_velocity=(self.densitydownstream_liquid/self.densitydownstream_vap)**(1/3)
            d=(1-self.x_forOrfice_thermogenic)/self.x_forOrfice_thermogenic
            self.downstreamvoidfraction=1/(1+d*self.slip_velocity*self.densitydownstream_vap/self.densitydownstream_liquid)
            self.mdotfalala_notvap=(1-self.downstreamvoidfraction)*self.mdot_spc+self.downstreamvoidfraction*self.mdot_hem_try2
            #self.mdotfalala_notvap2=(1-self.downstreamvoidfraction)*self.mdot_spc+self.downstreamvoidfraction*self.mdot_hem_try3
            self.mdotfalala_vap=(1-self.downstreamvoidfraction)*self.mdot_hem_try2+self.downstreamvoidfraction*self.mdot_spc
            self.MassFlowrateSpireapeat=self.MassFlowrateSpi
            self.mdotDryer=(self.MassFlowrateSpi+self.mdot_hem_try2)*0.5
            q1=DischargeCo_SPI*InjectorArea*self.Vapour_PressurePa/(np.sqrt(self.Oxtanktemp))
            q2=np.sqrt(self.gammanitrous/(self.zfactor_Gas*188.9))
            q3=(2*self.delta_enthalpy)/(self.zfactor_Gas*188.9*self.Temp_Injector*self.gammanitrous)
            q4=np.sqrt(q3)        
            q5=1+(((self.gammanitrous-1)/2)*q3)  
            q6=q5**(-((self.gammanitrous+1)/(2*(self.gammanitrous-1))))
            self.trymdot=q1*q2*q4*q6
            
            # Compressibility factor Z
            #try:
               # R_specific = 8.314462618 / CP.PropsSI('M', 'NitrousOxide')
                #rho_calc = CP.PropsSI('Dmass', 'T', t, 'P', p, 'NitrousOxide')
               # self.Z_factor_calc = p / (rho_calc * R_specific * t)
            #except:
             #   self.Z_factor_calc = 0.0
    
    def printshit(self):
            print("hello")
    def Calc_CD(self,chamberpressure_PA):
        
        if self.status=="vapour":
            self.Temp_Injector=PropsSI('T', 'P', chamberpressure_PA, 'S',self.realupstream_entropy, 'NitrousOxide')
            self.Den_forCD=PropsSI('D', 'P', chamberpressure_PA, 'S',self.realupstream_entropy, 'NitrousOxide')
            
        else:
            self.Temp_Injector=PropsSI('T', 'P', chamberpressure_PA, 'H',self.realupstream_enthalpy, 'NitrousOxide')
            self.Den_forCD=PropsSI('D', 'P', chamberpressure_PA, 'H',self.realupstream_enthalpy, 'NitrousOxide')
        self.InjectorVisc=self.refernce_viscosity*((self.Temp_Injector/self.refernce_temp)**1.5)*((self.refernce_temp+self.sutherland_nitrous)/(self.Temp_Injector+self.sutherland_nitrous))
        self.Reynoldsnumber_SPI=(self.velocityspi*self.injectorhydrolicdiameter*self.Den_forCD*0.66)/self.InjectorVisc
        self.Reynoldsnumber_HEM=(self.velocityhem*self.injectorhydrolicdiameter*self.Den_forCD*0.66)/self.InjectorVisc
        self.skinfriction_SPI=0.0791*(self.Reynoldsnumber_SPI**-0.25)
        self.skinfriction_HEM=0.0791*(self.Reynoldsnumber_HEM**-0.25)
        self.ExccessPdropCon_K=2.28 #this is a guess for now, should be from injector design
        self.CD_SPI= 1/(np.sqrt(4*self.skinfriction_SPI*self.LDratio+self.ExccessPdropCon_K))
        self.CD_HEM= 1/(np.sqrt(4*self.skinfriction_HEM*self.LDratio+self.ExccessPdropCon_K))


    


            

        



            



    
    def add_values(self):
        row = {}
        for attr in self._init_attrs:  
            val = getattr(self, attr)
            row[attr] = val
        self._output_rows.append(row)
    
    def finalize_output(self):
        """Convert buffered rows to DataFrame. Call this before accessing outputcsv."""
        if self._output_rows:
            self.outputcsv = pd.DataFrame(self._output_rows)
            self._output_rows = []
    #oxtank math notes-
    # figure out proper cheat
    #figure out temp during vapour phase
    #spec heat stuff (heat removed)
    #figure out 2 phase flow
#figure out boil delay
#figure out fill
#figure out ox tank heating
#figure out pressure losses calc
#gamma varying add

