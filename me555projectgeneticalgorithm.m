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


