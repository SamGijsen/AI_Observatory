%% AI Train-length setting: Two levels
% Implementation of the SBL-paradigm in a train-length setting.
% Level 1 of the model features a 'prediction'-action (at timepoint 1) that predicts the
% length up the upcoming stimulus train. It has a preference to observe it
% is correct (timepoint 'end'). A trial on level 1 is thus confined to one
% stimulus train. Level 2 features the regime progression; the fast- and
% slow-switching regimes are defined by their bias to emit shorter or
% longer trainlengths. Current implementation simplifies transition
% probabilities to alternation probabilities.

clear
rng(2)

ml = 4; % maximum train length, here determines also maximum time in trial
verbose = 0; % 1 to print a bunch of debugging info

% Lower level: Trains generate repetitions or alternations
D{1} = ones(ml,1); % true, unknown train length
D{2} = [1 zeros(1,ml+1)]'; % time in trial ('report', up-to-ml stimuli, 'delay')
D{3} = [1 zeros(1,ml)]'; % report: null, trainlengths (starts in state 'no report')
d = D;
d{2} = d{2}*1000; % prevent learning
d{3} = d{3}*1000;

% Likelihood mapping: specify type of stimulus at each time point that is
% expected under each sequence
% A{outcome modality}(outcome,factor1,factor2,...,factorN) 
% Outcome modality 1: Null, Repetitions, and alternations
for i = 1:ml+2 % ml+2-number of time points
    for j = 1:ml+1 % ml+1-number of choices (null + report on train length)
        % e.g. i=1,j=1 means first timepoint if one predicts 'length=1'
        temp = zeros(3,ml); % 3 outcomes by ml possible states (sequences)
        temp(1,:) = 1; % default setting is 'null'/no observation
        
        if i == 1 || i == ml+2 % no observation in report state
        else
            temp(1,i-1:end) = 0; % remove the flag for 'null' observation
            temp(2,i:end) = 1; % repetition if true train length is beyond time in trial (note i=2 is first obs)
            temp(3,i-1) = 1; % alternation if true train length matches time in trial
        end
        
        A{1}(:,:,i,j) = temp; % rows = outcomes (repeat, alternate), columns = states in each state factor
        
        if verbose
        if i < 60 % check output for first few timepoints
            if j < 60
                fprintf('time=%d choice=%d \n',i,j)
                temp
            end
        end
        end
        
    end
end

% Outcome modality 2: Feedback
% row 1 = 'no feedback', row 2 = correct, row 3 = '1 off', row 4 = '2 off', ..., row ml+1 = 'ml off'
% Feedback is always received upon final stimulus (for simplicity). 

for i = 1:ml+2 % ml+1-number of time points
    for j = 1:ml+1 % ml+1-number of choices (null + report on train length)
        % i=1,j=1 means first timepoint with the report 'length=1'
        
        temp = zeros(ml+1,ml); % ml+1 types of feedback by ml possible states (sequences)
        temp(1,:) = 1; % no feedback
        
        if i == 1 || j == 1
        else
            % provide feedback only in the final timepoint
            if i == ml+2
                % value depends on difference between report and true state
                % report = j-1, true state = temp(., :)
                temp(2,j-1) = 1;
                temp(1,j-1) = 0; % remove 'no feedback'
                % underestimation
                if (j-2)
                    for under = j-2 : -1 : 1
                    	temp(abs(under - (j-1)) + 2, under) = 1;
                        temp(1, under) = 0; % remove 'no feedback'
                    end
                end
                % overestimation
                if j <= ml
                for over = j:ml
                    temp(abs(over - (j-1)) + 2, over) = 1;
                    temp(1, over) = 0; % remove 'no feedback'
                end
                end
            end
        end
        
        A{2}(:,:,i,j) = temp; % save
        
        if verbose
        if i <100 % check output for first few timepoints
            fprintf('time=%d choice=%d \n',i,j)
            temp
        end
        end
        
    end
end

% Outcome modality 3: Prediction Observation: 1-to-1 mapping with
% prediction state
for i = 1:ml+2 % ml+1-number of time points
    for j = 1:ml+1 % ml+1-number of choices (null + report on train length)
        % i=1,j=1 means first timepoint with the report 'length=1'
        
        temp = zeros(ml+1,ml); % ml+1 types of prediction by ml possible states (sequences)
        % always maps to j
        temp(j,:) = 1;
        
        A{3}(:,:,i,j) = temp; % save
        
        if verbose
        if i < 100 % check output for first few timepoints
                i,j,temp
        end
        end

    end
end

% B{state factor}(state at time tau+1, state at time tau, action number)
B{1} = eye(ml); % train length is 'stable' within-trial
B{2} = eye(ml+3); % time in trial should progress forward, for this we want
% [ 0 0 0 0]
% [ 1 0 0 0]
% [ 0 1 0 0] 
% [ 0 0 1 1] (toy example for ml=2 thus times are [report obs1 obs2 delay])
B{2}(:,end) = [];
B{2}(2:end,:) = B{2}(1:end-1,:); % move I down
B{2}(end,:) = [];
B{2}(1,:) = zeros(1,ml+2); % top row are zeros
B{2}(end,end) = 1; % repeat delay phase (... right?)

% B{3} are controllable states. There are ml+1 actions: 'no report', or
% predict any train length.
% In terms of state transitions, this means the agent can move from no
% report to any other report but not back.
for i = 1:ml+1
    temp = eye(ml+1);
    temp(1,1) = 0; % never move to state 1
    temp(i,1) = 1; % from state 1 to state i
    B{3}(:,:,i) = temp;
end

% Policies V(time point, policy, state factor)
% ml-number of policies (column, rows correspond to timepoints)
T = ml+2; % number of timesteps
Nf = 3; % number of factors
Pi = ml; % number of policies
V = ones(T-1,Pi,Nf); % No actions regarding first two states

% For third state, we can report/predict any of the possible train lenghts
% at the first timepoint
V(1,:,3) = [2:ml+1];

% C{outcome modality}(outcome, time point)
% no preference for stimuli
C{1} = zeros(3,ml+2); % stimuli across timepoints
% preference to be correct - for this to work we need to always provide
% feedback at last timepoint. 
C{2} = zeros(ml+1, ml+2); % types of feedback across timepoints
C{2}(:,end) = [0 2*ml:-2:((2*ml)-2*(ml-1))]; %max reward of 2*ml, down stepwise by 2
% no preference for prediction per se
C{3} = zeros(ml+1,ml+2); % predictions across timepoints

mdp_1.T = T;
mdp_1.A = A;
%mdp_1.a = A;
mdp_1.B = B;
mdp_1.D = D;
mdp_1.d = D; % 'D' learning is on
mdp_1.C = C;
mdp_1.V = V;
OPTIONS.D = 1;

% Options labels, these are only here to make the SPM plots interpretable.
label.modality{1} = 'Observations'; label.outcome{1} = {'Null', 'Repetition', 'Alternation'};
label.modality{2} = 'Feedback'; label.outcome{2} = cell(1,ml+1);
label.outcome{2}{1} = 'No feedback';

label.factor{1} = 'Train Length';
label.factor{2} = 'Time in Trial';
label.factor{3} = 'Prediction';
label.name{1} = cell(1,size(D{1},1));
label.name{2} = cell(1, size(D{2},1));
label.name{2}{1} = 'Make Prediction';
label.name{2}{end} = 'Delay';
label.name{3} = cell(1, size(D{3},1));
label.name{3}{1} = 'No Prediction';
label.action{3} = cell(1, ml+1);
label.action{3}{1} = 'Stay';

for i = 1:ml
    label.outcome{2}{i+1} = [num2str(i-1) ' off'];
    label.name{1}{i} = ['train of ' num2str(i)];
    label.name{2}{i+1} = ['time ' num2str(i)];
    label.name{3}{i+1} = ['predict ' num2str(i)];
    label.action{3}{i+1} = ['predict ' num2str(i)];
end

mdp_1.label = label;
mdp_1.Aname = {'Stimulus', 'Feedback', 'Prediction Obs'};
mdp_1.Bname = {'Sequence', 'Time in Trial', 'Prediction'};
mdp_1.OPTIONS.D = 1;
% --- end of labeling

% TOP LEVEL ---------------------------------------------------------------
% Here, the slow and fast regimes emit train lengths with different
% probabilities
D2{1} = [1 1]'; % R1 vs R2 
d2 = D2;

% Outcomes are train lengths
% These approximate SBL-trainlengths from the study
% A2 = [0.12 0.25;
%       0.3 0.52;
%       0.18 0.15;
%       0.14 0.05;
%       0.08 0.02;
%       0.05 0.005;
%       0.04 0.004;
%       0.025 0.001;
%       0.02  0;
%       0.015 0;
%       0.015 0;
%       0.01  0;
%       0.005 0;
%       0.004 0;
%       0.003 0;
%       0.002 0;
%       0.002 0;
%       0.001 0;
%       0.001 0;
%       0.001 0];

A2{1} = [0.15 0.3;
      0.2 0.5;
      0.4 0.1;
      0.25 0.1];

% if we run across trials, use the identity matrix here
%B2{1} = eye(2); % regime doesnt change per trial
B2{1} = [0.9 0.1; 0.1 0.9];

mdp_1 = spm_MDP_check(mdp_1);
mdp.MDP = mdp_1;
% "identifies lower level state factors (rows) with higher level observation modalities (columns)"
% On lower level, first state factor is 'true trainlength', which is the
% 'observation' for higher level, thus [1;0;0]
mdp.link = [1;0;0]; 
mdp.A = A2;
%mdp.a = A2;
mdp.D = D2;
mdp.d = D2;
mdp.B = B2;

OPTIONS.D = 1;

mdp.Aname = {'Train length'};
mdp.Bname = {'Regime'};

N = 20; % number of trials
% We can run the top-level across-trials
% mdp.T = 1;
% MDP_N(1:N) = deal(mdp);
% for i = 1:N
%     MDP_N(i).o = 3;
% end
% for i = 6:N
%     MDP_N(i).o = 2;
% end

%... or we can run it within trial (thus, across 'outcomes')
mdp.T = 20;
% here we prespecify the observation sequence like we would analyzing the
% SBL study
mdp.o = [3 4 3 3 2 3 4 2 1 2 2 2 1 1 3 1 1 2 2 2];

MDP = spm_MDP_VB_X_tutorial(mdp);

%% Plotting Across OUTCOMES
p_r = NaN(2,N); % posterior over regimes
p_tl = NaN(ml,N); % posterior prior over trainlengths
p_u = NaN(ml+1,N); % posterior over actions
u = NaN(1,N); % action
o = NaN(1,N); % true trainlength
for i = 1:N
    p_tl(:,i) = MDP.mdp(i).d{1}/sum(MDP.mdp(i).d{1});
    p_u(:,i) = squeeze([MDP.mdp(i).P(1,1,:,1)]);
    u(i) = MDP.mdp(i).u(3,1);
    o(i) = MDP.o(i);
end

%p(regime)
subplot(4,1,1)
plot(MDP.Q{1}','linewidth',2)
leg = cell(1,2);
for i = 1:2
    leg{i} = sprintf('p(regime=%d)',i);
end
legend(leg)
xlim([0.5,N+0.5])
title('Smoothing distr. over Regimes')

% p(trainlength)
subplot(4,1,2)
plot(p_tl','linewidth',2)
leg = cell(1,ml);
for i = 1:ml
    leg{i} = sprintf('p(tl=%d)',i);
end
legend(leg)
xlim([0.5,N+0.5])
title('Posterior over Observations (TL)')

% p(action)
subplot(4,1,3)
plot(p_u(2:end,:)','linewidth',2)
leg = cell(1,ml);
for i = 1:ml
    leg{i} = sprintf('p(predict=%d)',i);
end
legend(leg)
xlim([0.5,N+0.5])
title('Posterior over Policies')

subplot(4,1,4)
scatter(1:N, u-1, 100,'filled', 'b'); hold on
scatter(1:N, o, 33,'filled', 'r');
legend('Predictions', 'Observations')
xlim([0.5,N+0.5])
title('Policies and Observations')

%% Plotting Across TRIALS
p_r = NaN(2,N); % posterior over regimes
p_tl = NaN(ml,N); % posterior prior over trainlengths
p_u = NaN(ml+1,N); % posterior over actions
u = NaN(1,N); % action
o = NaN(1,N); % true trainlength
for i = 1:N
    p_tl(:,i) = MDP(i).mdp.d{1}/sum(MDP(i).mdp.d{1});
    p_u(:,i) = squeeze([MDP(i).mdp.P(1,1,:,1)]);
    u(i) = MDP(i).mdp.u(3,1);
    o(i) = MDP(i).o;
end

% p(trainlength)
subplot(3,1,1)
plot(p_tl','linewidth',2)
leg = cell(1,ml);
for i = 1:ml
    leg{i} = sprintf('p(tl=%d)',i);
end
legend(leg)
xlim([0.5,N+0.5])
title('Posterior over Observations (TL)')

% p(action)
subplot(3,1,2)
plot(p_u(2:end,:)','linewidth',2)
leg = cell(1,ml);
for i = 1:ml
    leg{i} = sprintf('p(predict=%d)',i);
end
legend(leg)
xlim([0.5,N+0.5])
title('Posterior over Policies')

subplot(3,1,3)
scatter(1:N, u-1, 100,'filled', 'b'); hold on
scatter(1:N, o, 33,'filled', 'r');
legend('Predictions', 'Observations')
xlim([0.5,N+0.5])
title('Policies and Observations')

