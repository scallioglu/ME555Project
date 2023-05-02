clear all; clc;

p = 4; %%%% number of candidates
T = 0.2;
natom = 38; length=10;
candidates = rand(p,natom*3)*length-length/2;
all_candidates = candidates;

energy_values = [];
for i = 1:p
candidate = candidates(i,:);
[x, fval] = cg_minimization(candidate,natom); %%%%% find local minimum-energy positions
candidates(i,:) = x;
energy_values = [energy_values;fval];
end
all_energy_values = energy_values;

for generation = 1:500
    generation
    [parent1, parent2] = select_parents(candidates,T,natom);
    child = mate(parent1, parent2,natom);
    [child_candidate, energy_child] = cg_minimization(child,natom);     
    if energy_child < max(energy_values)
        [emax,indx] = max(energy_values);
        candidates(indx,:) = child_candidate;  
        energy_values(indx,:) = energy_child; 
        all_energy_values = [all_energy_values;energy_child];
        all_candidates = [all_candidates;child_candidate];
    end
    energy_values
end

moviefile = fopen('movie_38atoms.XYZ','w'); 
for gen = 1:size(all_candidates,1)
c = all_candidates(gen,:);
fprintf(moviefile,'%5i \n', natom);fprintf(moviefile,'Generation %5i',gen);
fprintf(moviefile,'\n');
   for i = 1:natom    
            fprintf(moviefile,'C %11.4f %11.4f %11.4f \n', c(1,3*i-2), c(1,3*i-1), c(1,3*i));
   end
end

[emin,ind]=min(all_energy_values);
final_candidate = all_candidates(ind);
moviefile = fopen('final_structure_38atom.XYZ','w'); 

fprintf(moviefile,'%5i \n', natom);fprintf(moviefile,'Final Structure');
fprintf(moviefile,'\n');
for i = 1:natom    
    fprintf(moviefile,'C %11.4f %11.4f %11.4f \n', c(1,3*i-2), c(1,3*i-1), c(1,3*i));
end

plot(1:size(all_energy_values,1),all_energy_values,'o');xlabel('Generation');
ylabel('Energy (eV/atom)')


function energy = calculate_energy(candidate,natom)

energy = 0;
for i = 1:natom-1
ri = candidate(1,3*i-2:3*i);
for j = i+1:natom
rj = candidate(1,3*j-2:3*j);
rij = norm(ri-rj);
elj = 1/rij^12-2/rij^6;
energy = energy+elj;
end
end
energy = energy/natom;
end

function grad = calculate_gradient(candidate,natom)

grad = zeros(1,3*natom);

for i = 1:natom-1
ri = candidate(1,3*i-2:3*i);
f = zeros(1,3);
for j = i+1:natom
rj = candidate(1,3*j-2:3*j);
rij = norm(ri-rj);
flj = -12/rij^13+12/rij^7;
fmag = flj;
f = fmag.*(ri-rj)./rij;
grad(1,3*i-2:3*i) = grad(1,3*i-2:3*i)+f;
grad(1,3*j-2:3*j) = grad(1,3*j-2:3*j)-f;
end
end

grad = grad/natom;
end

function [x, fval] = cg_minimization(x0,natom)
max_iter = 200;
tol = 1e-6;

% Initialize variables
x = x0;
fval = calculate_energy(x,natom);
grad = calculate_gradient(x,natom);
d = -grad;

% Iterate
for k = 1:max_iter
    % Calculate alpha using line search
    alpha = line_search(x,d,natom);
    
    % Update x and grad
    x_new = x+alpha*d;
    fval_new = calculate_energy(x_new,natom);
    grad_new = calculate_gradient(x_new,natom);
    
    % Check convergence
    if norm(grad_new) < tol
        break;
    end
    
    % Update the conjugate direction
    beta = max(0,dot(grad_new,grad_new-grad)/dot(grad,grad));
    d = -grad_new+beta*d;
    
    % Update variables for next iteration
    x = x_new;
    fval = fval_new;
    grad = grad_new;
end

end

function alpha = line_search(x,d,natom)
% Backtracking line search with Armijo rule and curvature
rho = 0.5;
c = 0.1;
max_iter = 100;

% Initialize variables
alpha = 1;
fval = calculate_energy(x,natom);
grad = calculate_gradient(x,natom);
count = 0;

% Iterate until the Armijo rule is met
while calculate_energy(x+alpha*d,natom) > fval+c*alpha*dot(grad,d) && count<max_iter
    alpha = rho*alpha;
    count = count + 1;
end
end


function [parent1, parent2] = select_parents(candidates,T,natom)
    % Calculate the energy of each candidate in the population
    energies = zeros(size(candidates,1), 1);
    for i = 1:size(candidates,1)
        energies(i) = calculate_energy(candidates(i,:),natom);
    end
    
    % Calculate the probability of selecting each candidate
    partition = sum(exp(-energies/T));
    probabilities = exp(-energies/T)/partition;
    
    % Select the parents based on the probabilities
    parent1 = candidates(find(rand<=cumsum(probabilities), 1),:);
    parent2 = candidates(find(rand<=cumsum(probabilities), 1),:);

    while size(parent1,1) == 0 ||  size(parent2,1) == 0 
        disp('Working on parents')
        cumsum(probabilities)
        parent1 = candidates(find(rand<=cumsum(probabilities), 1),:);
        parent2 = candidates(find(rand<=cumsum(probabilities), 1),:);
    end

end


function child = mate(parent1, parent2,natom)
    % Choose a random plane passing through the center of mass of each parent cluster
    parent1 = reshape(parent1,3,natom); parent1=parent1';
    parent2 = reshape(parent2,3,natom); parent2=parent2';

    center_of_mass1 = mean(parent1, 1);
    center_of_mass2 = mean(parent2, 1);
    normal = randn(1,3);
    normal = normal/norm(normal);
    plane_point1 = center_of_mass1 + randn(1, 3);
    plane_point2 = center_of_mass2 + randn(1, 3);
    [intersection_point1, intersection_point2] = intersect_plane_with_line(plane_point1, normal, parent1);
    [intersection_point3, intersection_point4] = intersect_plane_with_line(plane_point2, normal, parent2);
    
    % Assemble the child from the atoms of parent1 above the plane and parent2 below the plane
    child = [parent1(parent1(:,3) > intersection_point1(3),:); parent2(parent2(:,3) < intersection_point3(3),:)];
    
    % Check if the child has the correct number of atoms
    while size(child, 1) ~= size(parent1, 1)
        disp('Working on');
        disp('Child size = ')
        disp(size(child,1))
        % Translate parent1 and parent2 in opposing directions normal to the cut plane
        displacement = (size(parent1, 1)-size(child, 1))/2;
        parent1 = parent1+repmat(displacement*normal,size(parent1,1),1);
        parent2 = parent2-repmat(displacement*normal,size(parent2,1),1);
        
        % Recalculate intersection points and assemble the child again
        [intersection_point1, intersection_point2] = intersect_plane_with_line(plane_point1, normal, parent1);
        [intersection_point3, intersection_point4] = intersect_plane_with_line(plane_point2, normal, parent2);
        child = [parent1(parent1(:,3) > intersection_point1(3),:); parent2(parent2(:,3) < intersection_point3(3),:)];

        % Ensure that child size never exceeds natom
        if size(child, 1) > natom
            child = child(1:natom, :);
        else
            normal = randn(1,3);
            normal = normal/norm(normal);
        end

    end
    
    child = reshape(child',1,[]);
end

function [intersection_point1, intersection_point2] = intersect_plane_with_line(plane_point, normal,line_points)
    % Calculate the intersection point of a plane and a line
    line_vector = line_points(2,:)-line_points(1,:);
    line_vector = line_vector/norm(line_vector);
    t = dot(plane_point-line_points(1,:),normal)/dot(line_vector,normal);
    intersection_point1 = line_points(1,:)+t*line_vector;
    intersection_point2 = line_points(2,:)+t*line_vector;
end





% function forces = calculate_forces(candidate)
% natom = 60;
% energy = 0;
% forces = [];
% for i = 1:natom
% ri = candidate(1,3*i-2:3*i);
% x = 0;
% others = setdiff(1:natom,i);
% for j = 1:natom-1
% index = others(j);
% rj = candidate(1,3*index-2:3*index);
% r = norm(ri-rj);
% x = x+calculate_phi(r);
% end
% dx_dr = calculate_dxdr(r);
% dE_dr = -1.24251169551587*10^(-7)*4*x^3*dx_dr+2.3539221516757*10^(-5)*3*x^2*dx_dr-1.7896349903996*10^(-3)*2*x*dx_dr+0.5721151498619*dx_dr;
% fi = dE_dr.*(ri-rj)./r;
% forces = [forces,fi];
% end
% end
% 
% function dx_dr = calculate_dxdr(r)
% phi0 = 8.18555;
% m = 3.30304; mc = 8.6655;
% dc = 2.1052;
% d0 = 1.64; d1 = 2.57;
% dx_dr = 0;
% 
% if r < 2.6
% if r < d1
% exp_power = m*(-(r/dc)^mc+(d0/dc)^mc);
% dx_dr = -m*phi0*d0^m*exp(exp_power)/r^(m+1)-m*mc*r^(mc-1)*phi0*(d0/r)^m*exp(exp_power)/dc^mc;
% else
% dx_dr = 6.6024390226*10^(-5)*3*(r-d1)^2+2.1043303374*10^(-5)*(r-d1)*2-1.440864056*10^(-6);
% end
% else 
% dx_dr = 0;
% end
% end
% 
% function energy = calculate_energy(candidate)
% e_rep = calculate_erep(candidate);
% e_bs = calculate_ebs(candidate);
% energy = e_rep+e_bs;
% end
% 
% function e_rep = calculate_erep(candidate)
% natom = 60;
% energy = 0;
% for i = 1:natom
% ri = candidate(1,3*i-2:3*i);
% x = 0;
% others = setdiff(1:natom,i);
% for j = 1:natom-1
% index = others(j);
% rj = candidate(1,3*index-2:3*index);
% r = norm(ri-rj);
% x = x+calculate_phi(r);
% end
% f = -1.24251169551587*10^(-7)*x^4+2.3539221516757*10^(-5)*x^3-1.7896349903996*10^(-3)*x^2+0.5721151498619*x-2.5909765118191;
% energy = energy+f;
% end
% energy = energy/2; %%%% because of double counting
% e_rep = energy/natom;
% end
% 
% 
% function e_bs = calculate_ebs(candidate)
% natom = 60;
% energy = 0;
% E_s = -2.99; % eV
% E_p = 3.71; % eV
% V_sssigma = -5.0; % eV
% K_ps = 4.7; % eV
% V_ppsigma = 5.5; % eV
% V_pppi = -1.55; % eV
% 
% Eo=ones(1,natom).*(E_s + 2*E_p) / 3;
% H=diag(Eo);
% 
% for i = 1:natom
% ri = candidate(1,3*i-2:3*i);
% for j = i+1:natom
% rj = candidate(1,3*j-2:3*j);
% r = norm(ri-rj);
% s = calculate_s(r);
% 
% V_ss_ij = V_sssigma*s;
% V_pp_ij = V_ppsigma*(1-s);
% V_pppi_ij = V_pppi*(1-s)*3*r^2;
% H(i,j) = V_ss_ij+V_pp_ij+V_pppi_ij;
% H(j,i) = H(i,j); % Hamiltonian is symmetric
% end
% end
% 
% eigvals = eig(H);
% e_bs = sum(eigvals)/natom;
% end
% 
% 
% function s = calculate_s(r)
% rc = 2.18;
% n = 2; nc=6.5;
% r0=1.536329;
% r1 = 2.45;
% 
% if r < 2.6
% if r < r1
% exp_power = n*(-(r/rc)^nc+(r0/rc)^nc);
% s = (r0/r)^n*exp(exp_power);
% else
% ts = 0.3542874332380*(r-r1)^3+0.1932365259144*(r-r1)^2-8.1885359517898*10^(-2)*(r-r1)+6.7392620074314*10^(-3);
% s = ts;
% end
% else
% s = 0;
% end
% 
% end
% 
% function phi = calculate_phi(r)
% phi0 = 8.18555;
% m = 3.30304; mc = 8.6655;
% dc = 2.1052;
% d0 = 1.64; d1 = 2.57;
% 
% if r < 2.6
% if r < d1
% exp_power = m*(-(r/dc)^mc+(d0/dc)^mc);
% phi = phi0*(d0/r)^m*exp(exp_power);
% else
% tphi = 6.6024390226*10^(-5)*(r-d1)^3+2.1043303374*10^(-5)*(r-d1)^2-1.440864056*10^(-6)*(r-d1)+2.2504290109*10^(-8);
% phi = tphi;
% end
% else 
% phi = 0;
% end
% 
% end