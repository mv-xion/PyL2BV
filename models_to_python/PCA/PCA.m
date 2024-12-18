classdef PCA
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        lambda
        basis
        label_name={'Principal component analysis (PCA)' 'PCA' 1}
        clusterop=false;
        sigmaopt=false;
        wl
        indx
    end
    
    methods
        function this=PCA
        end
        function this=regression(this,Xp,Y,Nfeat,dummy,estimateSigmaMethod,test,useropts)
            if nargin==8
                N = size(Xp,2);
                Cxx  = (Xp'*Xp)/(N-1);
                [A,D] = eigs(Cxx,Nfeat);
                %sumvar=sum(diag(D));
                this.lambda=D;
                this.basis=A;
                %disp(['PCA Variance for each component: [' sprintf('%0.4f  ',100*(diag(D)./sumvar)) ']'])
            end
        end
        function out=reconst(this,out1)
            out = out1*pinv(this.basis);
        end
        function out=addMLRAplots(this,ext_,wl_)%GENERAL TODOS LOS MODELOS
            if exist('ext_','var')~=1
                ext_='output';
            end
            if exist('wl_','var')==1                
                this.wl=wl_;
            else
                this.wl=[];
            end  
            out= {deblank(['DR-PCA ' ext_]) @this.graficarpca};
        end
        function nvar=getcomp(this)
            nvar=size(this.lambda,1);
        end
        function graficarpca(this,model,opt_general)
            tipos={'Variance contribution (%)' @this.variance_contribution;...
                'PC value' @this.pca_value_plot;...
                'Cumulative absolute PC contibutions' @this.pca_cum_abs_plot};            
            pca_main_plots(this,model,tipos,opt_general)
        end
        function pca_value_plot(this,gui_options)
            coeff=this.basis;
            Wav=gui_options.wl;
            figure, clf
            plot(Wav,coeff(:,(1:gui_options.comp)),'linewidth',3)
            grid on, axis tight
            leyenda=cell(1,gui_options.comp);
            for i=1:gui_options.comp
                leyenda{i}=sprintf('PC #%d',i);
            end
            legend(leyenda,'location','northeast')
            xlabel('Wavelength'), ylabel('PC value')
            
            if gui_options.general.exportar
                fileID = fopen(gui_options.general.salida_archivo,'w');
                fprintf(fileID, '%s\n','PC Values (%)');
                fprintf(fileID,'%s','Wavelength');
                fprintf(fileID, ', PC #%d',1:gui_options.comp);
                fprintf(fileID,'\n');
                fclose(fileID);
                dlmwrite(gui_options.general.salida_archivo,[Wav,coeff(:,(1:gui_options.comp))],'delimiter',',','-append')
                fprintf('file: %s exported OK\n',gui_options.general.salida_archivo)
            end
            
        end
        function pca_cum_abs_plot(this,gui_options)
            
            coeff=cumsum(abs(this.basis),2);
            Wav=gui_options.wl;
            figure, clf
            plot(Wav,coeff(:,(1:gui_options.comp)),'linewidth',3)
            grid on, axis tight
            leyenda=cell(1,gui_options.comp);
            for i=1:gui_options.comp
                leyenda{i}=sprintf('Sum PC #%d',i);
            end
            legend(leyenda,'location','northeast')
            xlabel('Wavelength'), ylabel('Cumulative absolute PC contibutions')
            
            if gui_options.general.exportar
                fileID = fopen(gui_options.general.salida_archivo,'w');
                fprintf(fileID, '%s\n','Cumulative absolute PC contibutions (%)');
                fprintf(fileID,'%s','Wavelength');
                fprintf(fileID, ', PC #%d',1:gui_options.comp);
                fprintf(fileID,'\n');
                fclose(fileID);
                dlmwrite(gui_options.general.salida_archivo,[Wav,coeff(:,(1:gui_options.comp))],'delimiter',',','-append')
                fprintf('file: %s exported OK\n',gui_options.general.salida_archivo)
            end
            
        end
        
        function variance_contribution(this,gui_options)            
            
            figure
            var_contrib = 100*(diag(this.lambda)./sum(diag(this.lambda)));
            bar(var_contrib(1:gui_options.comp))
            xlabel('Principal component'), ylabel('Variance contribution (%)')
            hold on
            plot(cumsum(var_contrib(1:gui_options.comp)),'o-')
            hold off
            if gui_options.log
                set(gca,'yscale','log')
            end
            grid on
            if gui_options.general.exportar
                fileID = fopen(gui_options.general.salida_archivo,'w');
                fprintf(fileID, '%s\n','Variance contribution (%)');
                for i=1:gui_options.comp
                    fprintf(fileID, 'PC-%01d,   %f\n',i,var_contrib(i));
                end
                fclose(fileID);                
                fprintf('file: %s exported OK\n',gui_options.general.salida_archivo)
            end
        end
        function this=clasificacion(this,Xp,Y,Nfeat,dummy,estimateSigmaMethod,test,useropts)
            if nargin==8
                N = size(Xp,2);
                Cxx  = (Xp'*Xp)/(N-1);
                [A,D] = eigs(Cxx,Nfeat);
                %sumvar=sum(diag(D));
                this.lambda=D;
                this.basis=A;
                %disp(['PCA Variance for each component: [' sprintf('%0.4f  ',100*(diag(D)./sumvar)) ']'])
            end
        end
        function this=uclasificacion(this,Xp,Y,Nfeat,dummy,estimateSigmaMethod,test,useropts)
            if nargin==8
                N = size(Xp,2);
                Cxx  = (Xp'*Xp)/(N-1);
                [A,D] = eigs(Cxx,Nfeat);
                %sumvar=sum(diag(D));
                this.lambda=D;
                this.basis=A;
                %disp(['PCA Variance for each component: [' sprintf('%0.4f  ',100*(diag(D)./sumvar)) ']'])
            end
        end
        function Xt=project(this,X)
            Xt = X * this.basis; 
            if isprop(this,'indx')
                if ~isempty(this.indx)
                    Xt(:,~this.indx)=[];
                end
            end
        end
        function S=saveobj(this)
            for item=properties(this)'
                S.(item{1})=this.(item{1});
            end
        end
        function this=reload(this,S)
            if  isstruct(S)
                for item=fieldnames(S)'
                    this.(item{1})=S.(item{1});
                end
            end
        end
    end
    methods(Static)
        function obj=loadobj(S)
            obj=SIMFEAT_AL.PCA;
            obj=reload(obj,S);
        end
    end
end

