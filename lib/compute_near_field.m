function [saveName] = compute_near_field(args)
    arguments
        args.datapath string = '../raw_data/'
        args.include_id logical = false
        args.radius double = 2.35e-6
        args.n_particle double = 1.39
        args.n_medium double = 1.00
        args.wavelength double = 1064.0e-9
        args.NA double = 0.12
        args.xOffset double = 0.0e-6
        args.yOffset double = 0.0e-6
        args.zOffset double = 0.0e-6
        args.xSpan double = 20.0e-6
        args.zSpan double = 40.0e-6
        args.nx double = 101
        args.nz double = 101
        args.Nmax double = 50
        args.polarisation string = 'X'
        args.resimulate logical = true
    end

disp(' ');
disp('Running MATLAB simulation...');

if strcmp(args.datapath, '../raw_data/')
    args.include_id = true;
end

%%% Handle the arguments properly for both internal matlab execution
%%% as well as command-line execution where arguments are necessarily
%%% strings (probably printed by bash or equivalent)

wavelength_medium = args.wavelength / args.n_medium;

if strcmp(args.polarisation, 'X')
    polarisation = [1 0];
elseif strcmp(args.polarisation, 'Y')
    polarisation = [0 1];
else
    polarisation = [1 0];
end

saveFormatSpec = 'r%0.2fum_n%0.2f_na%0.3f_x%0.2f_y%0.2f_z%0.2f_Nmax%i';
saveName = strrep(sprintf(saveFormatSpec, args.radius*1e6, ...
                          args.n_particle, args.NA, ...
                          args.xOffset*1e6, args.yOffset*1e6, ...
                          args.zOffset*1e6, args.Nmax), '.', '_');

if args.include_id
    saveName = strcat(strip(args.datapath,'right','/'), '/', saveName);
else
    saveName = args.datapath;
end

%%% Check to see if the data exists yet. Make the folder if not. If yes,
%%% see whether the user wants to resimulate by examining the args
if not(isfolder(saveName))
    if not(args.resimulate)
        disp("Data doesn't exist to load and you requested NOT to resimulate");
        return
    end
    mkdir(saveName);
elseif not(args.resimulate)
    disp("Aborting simulation to load data from path:");
    disp(sprintf('    %s', saveName));
    return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Define the scatterer
T = ott.TmatrixMie(args.radius, 'wavelength0', args.wavelength, ...
                   'index_medium', args.n_medium, ...
                   'index_particle', args.n_particle, ...
                   'Nmax', args.Nmax);
               
%%% Unique Tmatrix for internal fields             
Tint = ott.TmatrixMie(args.radius, 'wavelength0', args.wavelength, ...
                      'index_medium', args.n_medium, ...
                      'index_particle', args.n_particle, ...
                      'internal', true, 'Nmax', args.Nmax);
                   
%%% Construct the input beam
ibeam = ott.BscPmGauss('NA', args.NA, 'polarisation', polarisation, ...
                       'index_medium', args.n_medium, ...
                       'wavelength0', args.wavelength, ...
                       'Nmax', args.Nmax);

ibeam = ibeam.translateXyz([args.xOffset; args.yOffset; -args.zOffset], ...
                           'Nmax', args.Nmax);

%%% LET THE SCATTERING BEGIN %%%
sbeam = T * ibeam;
intbeam = Tint * ibeam;

%%% Construct a "total" representation of the beam
totbeam = sbeam.totalField(ibeam);


%%% Translate back to the origin after doing the scattering, since the 
%%% nearfield imaging is aligned to the optical focus, not the scatterer
% ibeam = ibeam.translateXyz([-args.xOffset; -args.yOffset; +args.zOffset], ...
%                             'Nmax', args.Nmax);
% sbeam = sbeam.translateXyz([-args.xOffset; -args.yOffset; +args.zOffset], ...
%                             'Nmax', args.Nmax);
% totbeam = totbeam.translateXyz([-args.xOffset; -args.yOffset; +args.zOffset], ...
%                                 'Nmax', args.Nmax);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%     NEAR FIELD     %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Ensure a regular basis so the field is accurate in the vicinity
%%% of the scatterer itself
intbeam.basis = 'regular';
ibeam.basis = 'regular';
sbeam.basis = 'regular';
totbeam.basis = 'regular';

%%% Construct the points we want to sample
xpts = linspace(-0.5*args.xSpan, 0.5*args.xSpan, args.nx);
zpts = linspace(-0.5*args.zSpan, 0.5*args.zSpan, args.nz);

nearpts = zeros(3, args.nx*args.nz);
for i = 1:args.nx
    for k = 1:args.nz
        ind = (k-1)*args.nx + i;
        nearpts(:,ind) = [xpts(i);0;zpts(k)];
    end
end
        
%%% Sample the E and H fields at the requested points
[Ei, Hi] = ibeam.emFieldXyz(nearpts);
[Es, Hs] = sbeam.emFieldXyz(nearpts);

[Eint, Hint] = intbeam.emFieldXyz(nearpts);
[Etot, Htot] = totbeam.emFieldXyz(nearpts);

%%% Write all that shit to a few files
disp(' ')
disp('Writing data to:');
disp(sprintf('    %s', saveName));

formatSpec = '%s/nearfield_%s_%s.txt';

writematrix(nearpts, ...
            sprintf('%s/nearfield_points.txt', saveName));

writematrix(real(Ei), sprintf(formatSpec, saveName, ...
                                    'inc', 'real'));
writematrix(imag(Ei), sprintf(formatSpec, saveName, ...
                                    'inc', 'imag'));

writematrix(real(Es), sprintf(formatSpec, saveName, ...
                                    'scat', 'real'));
writematrix(imag(Es), sprintf(formatSpec, saveName, ...
                                    'scat', 'imag'));

writematrix(real(Eint), sprintf(formatSpec, saveName, ...
                                    'int', 'real'));
writematrix(imag(Eint), sprintf(formatSpec, saveName, ...
                                    'int', 'imag'));

writematrix(real(Etot), sprintf(formatSpec, saveName, ...
                                    'tot', 'real'));
writematrix(imag(Etot), sprintf(formatSpec, saveName, ...
                                    'tot', 'imag'));

end
