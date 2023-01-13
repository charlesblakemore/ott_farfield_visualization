function [saveName] = compute_axial_force(args)
    arguments
        args.datapath string = '../raw_data/'
        args.include_id logical = false
        args.radius double = 2.35e-6
        args.rho double = 1850.0
        args.n_particle double = 1.39
        args.n_medium double = 1.00
        args.wavelength double = 1064.0e-9
        args.NA double = 0.12
        args.xOffset double = 0.0e-6
        args.yOffset double = 0.0e-6
        args.zSpan double = 40.0e-6
        args.nz double = 101
        args.Nmax double = 50
        args.polarisation string = 'X'
        args.resimulate logical = true
    end

disp(' ');
disp('Running MATLAB simulation...');

%%% Define some numbers we'll need later
c = physconst('LightSpeed');
g = 9.8067;
mass = args.rho * (4.0/3.0) * pi * args.radius^3;

%%% If the user didn't specify something besides the default path,
%%% make sure to include some specific information in the save path
%%% to avoid overwriting things
if strcmp(args.datapath, '../raw_data/')
    args.include_id = true;
end

%%% Handle the arguments properly for both internal matlab execution
%%% as well as command-line execution where arguments are necessarily
%%% strings (probably typed in a bash terminal or equivalent)
wavelength_medium = args.wavelength / args.n_medium;

if strcmp(args.polarisation, 'X')
    polarisation = [1 0];
elseif strcmp(args.polarisation, 'Y')
    polarisation = [0 1];
else
    polarisation = [1 0];
end

saveFormatSpec = 'r%0.2fum_n%0.2f_na%0.3f_x%0.2f_y%0.2f_Nmax%i';
saveName = strrep(sprintf(saveFormatSpec, args.radius*1e6, ...
                          args.n_particle, args.NA, ...
                          args.xOffset*1e6, args.yOffset*1e6, args.Nmax), '.', '_');

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
                   
%%% Construct the input beam
ibeam = ott.BscPmGauss('NA', args.NA, 'polarisation', polarisation, ...
                       'index_medium', args.n_medium, ...
                       'wavelength0', args.wavelength, ...
                       'Nmax', args.Nmax);
ibeam.basis = 'regular';

%%% Build the array of points to sample, assumed to be parallel to
%%% the optical/z-axis, possibly with some offset
coords = [0;0;1] .* linspace(-0.5*args.zSpan, 0.5*args.zSpan, args.nz);
coords(1,:) = coords(1,:) + args.xOffset;
coords(2,:) = coords(2,:) + args.yOffset;

%%% Calculate the force
full_calculation = ott.forcetorque(ibeam, T, 'position', coords);
axial_efficiency = full_calculation(3,:) * args.n_medium / c;

%%% Write all that shit to a few files
disp(' ')
disp('Writing data to:');
disp(sprintf('    %s', saveName));

formatSpec = '%s/axial_force_%s.txt';

writematrix(coords, ...
            sprintf('%s/axial_force_points.txt', saveName));

writematrix(axial_efficiency, sprintf(formatSpec, saveName, ...
                                    'efficiency'));
writematrix(mass * g ./ axial_efficiency, sprintf(formatSpec, saveName, ...
                                    'levitation_power'));

end
