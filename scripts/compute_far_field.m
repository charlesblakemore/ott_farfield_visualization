function []=compute_far_field(datapath, args)
    arguments
        datapath string
        args.radius double = 2.35e-6
        args.n_particle double = 1.39
        args.n_medium double = 1.00
        args.wavelength double = 1064.0e-9
        args.NA double = 0.12
        args.xOffset double = 0.0e-6
        args.yOffset double = 0.0e-6
        args.zOffset double = 0.0e-6
        args.ntheta double = 101
        args.nphi double = 101
        args.Nmax double = 50
        args.polarisation string = 'X'
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

saveFormatSpec = 'r%0.2fum_n%0.2f_na%0.3f_x%0.2f_y%0.2f_z%0.2f';
saveName = strrep(sprintf(saveFormatSpec, args.radius*1e6, ...
                          args.n_particle, args.NA, ...
                          args.xOffset*1e6, args.yOffset*1e6, ...
                          args.zOffset*1e6), '.', '_');

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




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%     FAR FIELD     %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Intent is to visualize forward- and back-scattered light patterns to 
%%% better inform design of the imaging system, as well as operating point
%%% of MS within a trap

%%% Construct the points we want to sample over the full unit sphere
thetapts = linspace(0,pi,args.ntheta);
phipts = linspace(0,2*pi,args.nphi);

%%% Build up a 2x(ntheta*nphi) array of all sampled points since that's
%%% how the OTT utility function samples the electric field
farpts = zeros(2, args.ntheta*args.nphi);
for i = 1:args.ntheta
    for k = 1:args.nphi
        ind = (k-1)*args.ntheta + i;
        farpts(:,ind) = [thetapts(i);phipts(k)];
    end
end


%%% Ensure we're using the outgoing basis representation for non-vanishing
%%% far-fields (since we have to regularize for the singularity at the 
%%% origin in the actual scattering problem)
ibeam.basis = 'outgoing';
sbeam.basis = 'outgoing';
totbeam.basis = 'outgoing';


%%% Sample the fieds at the desired points
[Ei_far, Hi_far] = ibeam.farfield(farpts(1,:),farpts(2,:));
[Es_far, Hs_far] = sbeam.farfield(farpts(1,:),farpts(2,:));
[Et_far, Ht_far] = totbeam.farfield(farpts(1,:),farpts(2,:));

%%% Write all that shit to a few files
mkdir('../raw_data', saveName)

formatSpec = '../raw_data/%s/farfield_%s_%s.txt';

writematrix(farpts, sprintf('../raw_data/%s/farfield_points.txt', saveName));

writematrix(real(Ei_far), sprintf(formatSpec, saveName, 'inc', 'real'));
writematrix(imag(Ei_far), sprintf(formatSpec, saveName, 'inc', 'imag'));

writematrix(real(Es_far), sprintf(formatSpec, saveName, 'scat', 'real'));
writematrix(imag(Es_far), sprintf(formatSpec, saveName, 'scat', 'imag'));

writematrix(real(Et_far), sprintf(formatSpec, saveName, 'tot', 'real'));
writematrix(imag(Et_far), sprintf(formatSpec, saveName, 'tot', 'imag'));





end
