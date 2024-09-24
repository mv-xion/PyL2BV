function [out] = GPR_ARTMO_to_GEE_Python_unc(model_mat_f,suffix,dir_out,dest)
% Description:
% model_mat_f: the model you want to run the script on
% suffix: default: ''
% dir_out: output directoriy you want to put the result file, default: 
%   the directory of your model
% dest: the destination you want to run your model on ('gee','py','py_gee')

prec_digits = 15;

[mat_dir,mat_f,~] = fileparts(model_mat_f);
if ~exist('suffix','var');suffix ='';end
if ~exist('dir_out','var')
    dir_out = fullfile(mat_dir,mat_f);
    if ~exist(dir_out,'dir');mkdir(mat_dir,mat_f);end
end
if ~exist('dest','var');dest = 'gee'; end

model   = load(model_mat_f);

% Edit for PCA
ispca = isfield(model.modelo.model.pca, 'model');
if ispca
    % temporary solution with chime code
    addpath(genpath('E:\CHIME\chime\auxiliar_class\ipl_tools\'));
    %tmp=SIMFEAT_AL.PCA
    pca_mat = model.modelo.model.pca.model.basis;
    disp('PCA exists');
end
wave_length = model.modelo.model.WaveLength;
% END of Edit 

% Getting model attributes
[~,modelFileName,~] = fileparts(model_mat_f);
name      = model.modelo.name;
units     = model.modelo.units;
model_full_name = model.modelo.name_model;
model_type = model.modelo.name_model_fcn;



% Collecting the model parameters
L_mat   = model.modelo.model.model.L;
Linv_mat   = inv(L_mat);

Xtr     = model.modelo.model.model.Xtrain;
alpha   = model.modelo.model.model.alpha;
mx      = model.modelo.model.normvalues.mx;
sx      = model.modelo.model.normvalues.sx;
mean    = model.modelo.model.mean;
hyp_vec = model.modelo.model.model.loghyper;
Bands   = numel(hyp_vec)-2;

invell2_v  = 1./exp(2*hyp_vec(1:Bands));
D_ell2  = diag(invell2_v);
sf2     = exp(2*hyp_vec(Bands+1));
s2      = exp(2*hyp_vec(Bands+2));
XTDX    = sum(Xtr*D_ell2.*Xtr,2);
s2unc   = s2 + sf2;

L_mat_f   = fullfile(dir_out,[mat_f '_L.txt']);
Linv_mat_f   = fullfile(dir_out,[mat_f '_Linv.txt']);
Xtr_f     = fullfile(dir_out,[mat_f '_Xtr.txt']);
alpha_f   = fullfile(dir_out,[mat_f '_alpha.txt']);
mx_f      = fullfile(dir_out,[mat_f '_mx.txt']);
sx_f      = fullfile(dir_out,[mat_f '_sx.txt']);
mean_f    = fullfile(dir_out,[mat_f '_mean.txt']);
inv_l2_f  = fullfile(dir_out,[mat_f '_inv_ell2.txt']);
sf2_f     = fullfile(dir_out,[mat_f '_sf2.txt']);
s2_f      = fullfile(dir_out,[mat_f '_s2.txt']);
XTDX_f    = fullfile(dir_out,[mat_f '_XTDX.txt']);
s2unc_f      = fullfile(dir_out,[mat_f '_s2unc.txt']);
% Edit for PCA
if ispca
    pca_mat_f     = fullfile(dir_out,[mat_f '_pca.txt']);
end
wave_length_f = fullfile(dir_out,[mat_f '_wave_length.txt']);
% END of Edit
Xtr_1 = reshape(Xtr,size(Xtr,1),1,size(Xtr,2));

dlmwrite(L_mat_f,  L_mat ,   'delimiter', ',', 'precision', prec_digits);
dlmwrite(Xtr_f,    Xtr,      'delimiter', ',', 'precision', prec_digits);
dlmwrite(alpha_f,  alpha,    'delimiter', ',', 'precision', prec_digits);
dlmwrite(mx_f,     mx,       'delimiter', ',', 'precision', prec_digits);
dlmwrite(sx_f,     sx,       'delimiter', ',', 'precision', prec_digits);
dlmwrite(mean_f,   mean,     'delimiter', ',', 'precision', prec_digits);
dlmwrite(inv_l2_f, invell2_v,'delimiter', ',', 'precision', prec_digits);
dlmwrite(sf2_f,    sf2,      'delimiter', ',', 'precision', prec_digits);
dlmwrite(s2_f,     s2,       'delimiter', ',', 'precision', prec_digits);
dlmwrite(XTDX_f,   XTDX,     'delimiter', ',', 'precision', prec_digits);
dlmwrite(s2unc_f,   s2unc,     'delimiter', ',', 'precision', prec_digits);
dlmwrite(Linv_mat_f,  Linv_mat ,   'delimiter', ',', 'precision', prec_digits);
% Edit for PCA
if ispca
    dlmwrite(pca_mat_f,  pca_mat ,   'delimiter', ',', 'precision', prec_digits);
end
dlmwrite(wave_length_f,  wave_length ,   'delimiter', ',', 'precision', prec_digits);
% END of Edit
out = 1;

switch dest
    case 'gee'
        Xtr_str     = matrix2str(Xtr,        ['var X_train' suffix ' = ee.Array([']            ,'['  ,''  ,','    , ']'   ,',' ,prec_digits ,']);' );
        alpha_str   = matrix2str(alpha',     ['var alpha_coefficients' suffix ' = ee.Image([['] ,''   ,''  ,','    , ''    ,''  ,prec_digits ,']]);' );
        mx_str      = matrix2str(mx,         ['var mx' suffix '       =  ee.Image([[']          ,''   ,''  ,','    , ''    ,''  ,prec_digits ,']]);' );
        sx_str      = matrix2str(sx,         ['var sx' suffix '       =  ee.Image([[']          ,''   ,''  ,','    , ''    ,''  ,prec_digits ,']]);' );
        mean_str    = matrix2str(mean,       ['var mean_model' suffix ' = ']                    ,''   ,''  ,','    , ''    ,''  ,prec_digits ,';' );
        inv_l2_str  = matrix2str(invell2_v', ['var hyp_ell' suffix '  = ee.Image([']            ,''   ,''  ,','    , ''    ,''  ,prec_digits ,']);' );
        sf2_str     = matrix2str(sf2,        ['var hyp_sig' suffix '  =']                       ,''   ,''  ,','    , ''    ,''  ,prec_digits ,';'   );
        s2_str      = matrix2str(s2,         ['var hyp_sign' suffix ' = ee.Array([']            ,''   ,''  ,''     , ''    ,''  ,prec_digits ,']);'   );
        XTDX_str    = matrix2str(XTDX,       ['var XDX_pre_calc' suffix ' =  ee.Image([']       ,'['   ,'[' , '],'  , ']'  ,','  ,prec_digits ,']);' );
        Linv_str    = matrix2str(Linv_mat,   ['var Linv_pre_calc' suffix ' =  ee.Array([']      ,'['   ,'' , ','  , ']'  ,','  ,prec_digits ,']);' );
        s2unc_str      = matrix2str(s2unc,   ['var hyp_sig_unc' suffix ' = ']                   ,''   ,''  ,','    , ''    ,''  ,prec_digits ,';'   );

        overall_model_f = fullfile(dir_out,'overall_model_gee.txt');

    case 'py'
        Xtr_str     = matrix2str(Xtr,        ['X_train_GREEN' suffix ' = np.array([']            ,'['  ,''  ,','    , ']'   ,',' ,prec_digits ,']);' );
        alpha_str   = matrix2str(alpha',     ['alpha_coefficients_GREEN' suffix ' = np.array([['] ,''   ,''  ,','    , ''    ,''  ,prec_digits ,']]);' );
        mx_str      = matrix2str(mx,         ['mx_GREEN' suffix '       =  np.array([[']          ,''   ,''  ,','    , ''    ,''  ,prec_digits ,']]);' );
        sx_str      = matrix2str(sx,         ['sx_GREEN' suffix '       =  np.array([[']          ,''   ,''  ,','    , ''    ,''  ,prec_digits ,']]);' );
        mean_str    = matrix2str(mean,       ['mean_model_GREEN' suffix ' = ']                    ,''   ,''  ,','    , ''    ,''  ,prec_digits ,';' );
        inv_l2_str  = matrix2str(invell2_v', ['hyp_ell_GREEN' suffix '  = np.array([']            ,''   ,''  ,','    , ''    ,''  ,prec_digits ,']);' );
        sf2_str     = matrix2str(sf2,        ['hyp_sig_GREEN' suffix '  =']                       ,''   ,''  ,','    , ''    ,''  ,prec_digits ,';'   );
        s2_str      = matrix2str(s2,         ['hyp_sign_GREEN' suffix ' = np.array([']            ,''   ,''  ,''     , ''    ,''  ,prec_digits ,']);'   );
        XTDX_str    = matrix2str(XTDX,       ['XDX_pre_calc_GREEN' suffix ' =  np.array([']       ,'['   ,'[' , '],'  , ']'  ,','  ,prec_digits ,']);' );
        Linv_str    = matrix2str(Linv_mat,   ['Linv_pre_calc_GREEN' suffix ' =  np.array([']      ,'['   ,'' , ','  , ']'  ,','  ,prec_digits ,']);' );
        s2unc_str      = matrix2str(s2unc,   ['hyp_sig_unc_GREEN' suffix ' = ']                   ,''   ,''  ,','    , ''    ,''  ,prec_digits ,';'   );     
        % Edit for PCA
        if ispca
            pca_str     = matrix2str(pca_mat,    ['pca_mat' suffix ' = np.array([']            ,'['  ,''  ,','    , ']'   ,',' ,prec_digits ,']);' );
        end
        wave_length_str = matrix2str(wave_length', ['wave_length' suffix '  = np.array([']            ,''   ,''  ,','    , ''    ,''  ,prec_digits ,']);' );
        %END of Edit
        overall_model_f = fullfile(dir_out,strcat(modelFileName, '.py'));

    case 'py_gee'
        Xtr_str     = matrix2str(Xtr,        ['X_train_GREEN' suffix ' = ee.Array([']            ,'['  ,''  ,','    , ']'   ,',' ,prec_digits ,']);' );
        alpha_str   = matrix2str(alpha',     ['alpha_coefficients_GREEN' suffix ' = ee.Image([['] ,''   ,''  ,','    , ''    ,''  ,prec_digits ,']]);' );
        mx_str      = matrix2str(mx,         ['mx_GREEN' suffix '       =  ee.Image([[']          ,''   ,''  ,','    , ''    ,''  ,prec_digits ,']]);' );
        sx_str      = matrix2str(sx,         ['sx_GREEN' suffix '       =  ee.Image([[']          ,''   ,''  ,','    , ''    ,''  ,prec_digits ,']]);' );
        mean_str    = matrix2str(mean,       ['mean_model_GREEN' suffix ' = ']                    ,''   ,''  ,','    , ''    ,''  ,prec_digits ,';' );
        inv_l2_str  = matrix2str(invell2_v', ['hyp_ell_GREEN' suffix '  = ee.Image([']            ,''   ,''  ,','    , ''    ,''  ,prec_digits ,']);' );
        sf2_str     = matrix2str(sf2,        ['hyp_sig_GREEN' suffix '  =']                       ,''   ,''  ,','    , ''    ,''  ,prec_digits ,';'   );
        s2_str      = matrix2str(s2,         ['hyp_sign_GREEN' suffix ' = ee.Array([']            ,''   ,''  ,''     , ''    ,''  ,prec_digits ,']);'   );
        XTDX_str    = matrix2str(XTDX,       ['XDX_pre_calc_GREEN' suffix ' =  ee.Image([']       ,'['   ,'[' , '],'  , ']'  ,','  ,prec_digits ,']);' );
        Linv_str    = matrix2str(Linv_mat,   ['Linv_pre_calc_GREEN' suffix ' =  ee.Array([']      ,'['   ,'' , ','  , ']'  ,','  ,prec_digits ,']);' );
        s2unc_str      = matrix2str(s2unc,   ['hyp_sig_unc_GREEN' suffix ' = ']                   ,''   ,''  ,','    , ''    ,''  ,prec_digits ,';'   );

        overall_model_f = fullfile(dir_out,'overall_model_python_to_gee.py');
end


Nmax_col = numel(Xtr_str);
fid = fopen(overall_model_f,'wt');
if fid
    switch dest
        case 'gee'
            cylinder
        case 'py'
            fprintf(fid,"import numpy as np\n\n");
        case 'py_gee'
            fprintf(fid,"import ee\n\nee.Initialize()\n\n");
    end
    split_str4file(fid,Xtr_str,Nmax_col);fprintf(fid,'\n');
    split_str4file(fid,alpha_str,Nmax_col);fprintf(fid,'\n');
    split_str4file(fid,mx_str,Nmax_col);fprintf(fid,'\n');
    split_str4file(fid,sx_str,Nmax_col);fprintf(fid,'\n');
    split_str4file(fid,mean_str,Nmax_col);fprintf(fid,'\n');
    split_str4file(fid,inv_l2_str,Nmax_col);fprintf(fid,'\n');
    split_str4file(fid,s2_str,Nmax_col);fprintf(fid,'\n');
    split_str4file(fid,sf2_str,Nmax_col);fprintf(fid,'\n');
    split_str4file(fid,s2unc_str,Nmax_col);fprintf(fid,'\n');
    if strcmp(dest,'py_gee')
        fprintf(fid,"XTrain_dim_GREEN=X_train_GREEN.length().toList().get(0).getInfo()\n\n");
    end
    split_str4file(fid,XTDX_str,Nmax_col);fprintf(fid,'\n');
    split_str4file(fid,Linv_str,Nmax_col);fprintf(fid,'\n');
    % Edit for PCA
    if ispca
        split_str4file(fid,pca_str,Nmax_col);fprintf(fid,'\n');
    end
    split_str4file(fid,wave_length_str,Nmax_col);fprintf(fid,'\n');
    % END of Edit
    fprintf(fid,"veg_index = '%s' ;\n",name);
    fprintf(fid,"units = '%s' ;\n",units);
    fprintf(fid,"model_type = '%s' ;\n",model_type);
    fprintf(fid,"model_name = '%s' ;\n",model_full_name);
    fprintf(fid,"model = '%s' ;\n",modelFileName);
    fprintf(fid,'\n');
    
end
fclose(fid);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper functions for the script %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function split_str4file(fid,str_info,Nmax_col)
    Ncol = numel(str_info);
    col_vec = 1:Nmax_col:Ncol;
    if col_vec(end)<Ncol
        col_vec=[col_vec Ncol+1];
    end
    
    for i_str=2:numel(col_vec)-1
        ind = col_vec(i_str);
        if ~strcmp(str_info(ind),',')
            cnd =true;
            while(cnd)
                ind=ind+1;
                if strcmp(str_info(ind),',')
                    cnd=false;
                end
            end
        end
        col_vec(i_str)=ind;
    end
       
    for i_str=1:numel(col_vec)-1
        fprintf(fid,'%s\n',str_info(col_vec(i_str):col_vec(i_str+1)-1));
    end



function out_str = matrix2str(matrix,header_str,st_line,delim_st,delim_end, end_line,delim_line,precision,tail_str)

% matrix     : Matrix to be prepared for GEE
% header_str : header string
% delim_st   : front element delimiter
% delim_end  : back element delimiter
% end_line   : end-line delimiter
% precision  : precision for number to string conversion
% tail_str   : string final tail

[nl,nc]   = size(matrix);
str_vec_l = header_str;
for i_l =1:nl
    str_vec_c = st_line;
    for i_c =1:nc
      if i_c < nc; delim_st_ =delim_st; else delim_st_ = '';  end

      if i_c < nc; delim_end_ =delim_end; else delim_end_ = '';  end
      str_vec_c=[str_vec_c delim_st_ num2str(matrix(i_l,i_c),precision) delim_end_];
    end
    if i_l<nl
        str_vec_c =[str_vec_c  end_line delim_line];
    else
        str_vec_c =[str_vec_c  end_line];
    end
    str_vec_l =  [str_vec_l str_vec_c];
end

out_str = [str_vec_l tail_str];