#!/usr/bin/env python
# coding: utf-8

# In[18]:


import torch
from collections.abc import Iterator

class DataGetter:
    """A pickleable mock-up of a Python iterator on a torch.utils.Dataloader."""

    def __init__(self, X: torch.Tensor, batch_size: int, seed: int = GLOBAL_SEED) -> None:
        """Calls the _init_data method on intialization of a DataGetter object."""
        torch.manual_seed(seed)
        self.X = X
        self.batch_size = batch_size
        self.data = []
        self._init_data(
            iter(torch.utils.data.DataLoader(self.X, batch_size=self.batch_size, shuffle=True))
        )

    def _init_data(self, iterator: Iterator) -> None:
        """Load all of the iterator into a list."""
        x = next(iterator, None)
        while x is not None:
            self.data.append(x)
            x = next(iterator, None)

    def __next__(self) -> tuple:
        """Analogous behaviour to the native Python next() but calling the
        .pop() of the data attribute.
        """
        try:
            return self.data.pop()
        except IndexError:  # Caught when the data set runs out of elements
            self._init_data(
                iter(torch.utils.data.DataLoader(self.X, batch_size=self.batch_size, shuffle=True))
            )
            return self.data.pop()


# In[23]:


GLOBAL_SEED = 24
torch.manual_seed(GLOBAL_SEED)
torch.set_default_tensor_type(torch.DoubleTensor)

def generate_normal_time_series_set(
    p: int, num_series: int, noise_amp: float, t_init: float, t_end: float, seed: int = GLOBAL_SEED
) -> tuple:
    """Generate a normal time series data set where each of the p elements of each sequence are 
    drawn from a normal distribution x_t ~ N(0, noise_amp)"""
    torch.manual_seed(seed)
    X = torch.normal(0, noise_amp, (num_series, p))
    T = torch.linspace(t_init, t_end, p)
    return X, T

def generate_anomalous_time_series_set(
    p: int,
    num_series: int,
    noise_amp: float,
    spike_amp: float,
    max_duration: int,
    t_init: float,
    t_end: float,
    seed: int = GLOBAL_SEED,
) -> tuple:
    """Generate an anomalous time series data set where the p elements of each sequence are
    drawn from a normal distribution x_t ~ N(0, noise_amp)."""
    torch.manual_seed(seed)
    Y = torch.normal(0, noise_amp, (num_series, p))
    for y in Y:
        # 5–10 spikes allowed
        spike_num = torch.randint(low=5, high=10, size=())
        durations = torch.randint(low=1, high=max_duration, size=(spike_num,))
        spike_start_idxs = torch.randperm(p - max_duration)[:spike_num]
        for start_idx, duration in zip(spike_start_idxs, durations):
            y[start_idx : start_idx + duration] += torch.normal(0.0, spike_amp, (duration,))
    T = torch.linspace(t_init, t_end, p)
    return Y, T


# In[24]:


import matplotlib.pyplot as plt

X_norm, T_norm = generate_normal_time_series_set(25, 25, 0.1, 0.1, 2 * torch.pi)
Y_anom, T_anom = generate_anomalous_time_series_set(25, 25, 0.1, 0.4, 5, 0, 2 * torch.pi)

plt.figure()
plt.plot(T_norm, X_norm[0], label="Normal")
plt.plot(T_anom, Y_anom[1], label="Anomalous")
plt.ylabel("$y(t)$")
plt.xlabel("t")
plt.grid()
leg = plt.legend()


# In[25]:


def make_atomized_training_set(X: torch.Tensor, T: torch.Tensor) -> list:
    """Convert input time series data provided to atomized tuple chunks: (xt, t)."""
    X_flat = torch.flatten(X)
    T_flat = T.repeat(X.size()[0])
    atomized = [(xt, t) for xt, t in zip(X_flat, T_flat)]
    return atomized


# In[26]:


def get_training_cycler(Xtr: torch.Tensor, batch_size: int, seed: int = GLOBAL_SEED) -> DataGetter:
    """Get an instance of the DataGetter class defined above"""
    return DataGetter(Xtr, batch_size, seed)


# In[27]:


import pennylane as qml
from itertools import combinations

def D(gamma: torch.Tensor, n_qubits: int, k: int = None, get_probs: bool = False) -> None:
    """Generates an n_qubit quantum circuit according to a k-local Walsh operator
    expansion. Here, k-local means that 1 <= k <= n of the n qubits can interact.
    See <https://doi.org/10.1088/1367-2630/16/3/033040> for more
    details. Optionally return probabilities of bit strings.
    """
    if k is None:
        k = n_qubits
    cnt = 0
    for i in range(1, k + 1):
        for comb in combinations(range(n_qubits), i):
            if len(comb) == 1:
                qml.RZ(gamma[cnt], wires=[comb[0]])
                cnt += 1
            elif len(comb) > 1:
                cnots = [comb[i : i + 2] for i in range(len(comb) - 1)]
                for j in cnots:
                    qml.CNOT(wires=j)
                qml.RZ(gamma[cnt], wires=[comb[-1]])
                cnt += 1
                for j in cnots[::-1]:
                    qml.CNOT(wires=j)
    if get_probs:
        return qml.probs(wires=range(n_qubits))


# In[30]:


n_qubits = 1
dev = qml.device("default.qubit", wires=n_qubits, shots=None)
D_one_qubit = qml.qnode(dev)(D)
_ = qml.draw_mpl(D_one_qubit, decimals=2)(torch.tensor([1, 0]), 1, 1, True)
_ = qml.draw_mpl(D_one_qubit, decimals=2)(torch.tensor([1, 0, 1, 0]), 3, 1, True)


# In[132]:


@qml.qnode(dev, interface="torch", diff_method="backprop")
def get_probs(
    xt: torch.Tensor,
    t: float,
    alpha: torch.Tensor,
    gamma: torch.Tensor,
    k: int,
    U: callable,
    W: callable,
    D: callable,
    n_qubits: int,
) -> torch.Tensor:
    """Measure the probabilities for measuring each bitstring after applying a
    circuit of the form W†DWU to the |0⟩^(⊗n) state. This
    function is defined for individual sequence elements xt.
    """
    U(xt, wires=range(n_qubits))
    W(alpha, wires=range(n_qubits))
    D(gamma * t, n_qubits, k)
    qml.adjoint(W)(alpha, wires=range(n_qubits))
    qml.DepolarizingChannel(0.5, wires=0)
    return qml.probs(range(n_qubits))


# In[133]:


def get_callable_projector_func(
    k: int, U: callable, W: callable, D: callable, n_qubits: int, probs_func: callable
) -> callable:
    """Using get_probs() above, take only the probability of measuring the
    bitstring of all zeroes (i.e, take the projector
    |0⟩^(⊗n)⟨0|^(⊗n)) on the time devolved state.
    """
    callable_proj = lambda xt, t, alpha, gamma: probs_func(
        xt, t, alpha, gamma, k, U, W, D, n_qubits
    )[0]
    
    return callable_proj


# In[134]:


def F(
    callable_proj: callable,
    xt: torch.Tensor,
    t: float,
    alpha: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    gamma_length: int,
    n_samples: int,
) -> torch.Tensor:
    """Take the classical expecation value of of the projector on zero sampling."""
    # length of gamma should not exceed 2^n - 1
    gammas = sigma.abs() * torch.randn((n_samples, gamma_length)) + mu
    expectation = torch.empty(n_samples)
    for i, gamma in enumerate(gammas):
        expectation[i] = callable_proj(xt, t, alpha, gamma)
    return expectation.mean()


# In[135]:


def callable_arctan_penalty(tau: float) -> callable:
    """Create a callable arctan function with a single hyperparameter
    tau to penalize large entries of sigma.
    """
    prefac = 1 / (torch.pi)
    callable_pen = lambda sigma: prefac * torch.arctan(2 * torch.pi * tau * sigma.abs()).mean()
    return callable_pen


# In[136]:


def get_loss(
    callable_proj: callable,
    batch: torch.Tensor,
    alpha: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    gamma_length: int,
    n_samples: int,
    callable_penalty: callable,
) -> torch.Tensor:
    """Evaluate the loss function ℒ, defined in the background section
    for a certain set of parameters.
    """
    X_batch, T_batch = batch
    loss = torch.empty(X_batch.size()[0])
    for i in range(X_batch.size()[0]):
        # unsqueeze required for tensor to have the correct dimension for PennyLane templates
        loss[i] = (
            1
            - F(
                callable_proj,
                X_batch[i].unsqueeze(0),
                T_batch[i].unsqueeze(0),
                alpha,
                mu,
                sigma,
                gamma_length,
                n_samples,
            )
        ).square()
    return 0.5 * loss.mean() + callable_penalty(sigma)


# In[137]:


def get_initial_parameters(
    W: callable, W_layers: int, n_qubits: int, seed: int = GLOBAL_SEED
) -> dict:
    """Randomly generate initial parameters."""
    torch.manual_seed(seed)
    init_alpha = torch.rand(W.shape(W_layers, n_qubits))
    init_mu = torch.rand(1)
    # Best to start sigma small and expand if needed
    init_sigma = torch.rand(1)
    init_params = {
        "alpha": (2 * torch.pi * init_alpha).clone().detach().requires_grad_(True),
        "mu": (2 * torch.pi * init_mu).clone().detach().requires_grad_(True),
        "sigma": (0.1 * init_sigma + 0.05).clone().detach().requires_grad_(True),
    }
    return init_params


# In[138]:


def train_model_gradients(
    lr: float,
    init_params: dict,
    pytorch_optimizer: callable,
    cycler: DataGetter,
    n_samples: int,
    callable_penalty: callable,
    batch_iterations: int,
    callable_proj: callable,
    gamma_length: int,
    seed=GLOBAL_SEED,
    print_intermediate=False,
) -> dict:
    """Train the QVR model (minimize the loss function)."""
    torch.manual_seed(seed)
    opt = pytorch_optimizer(init_params.values(), lr=lr)
    alpha = init_params["alpha"]
    mu = init_params["mu"]
    sigma = init_params["sigma"]

    def closure():
        opt.zero_grad()
        loss = get_loss(
            callable_proj, next(cycler), alpha, mu, sigma, gamma_length, n_samples, callable_penalty
        )
        loss.backward()
        return loss

    loss_history = []
    for i in range(batch_iterations):
        loss = opt.step(closure)
        loss_history.append(loss.item())
        if batch_iterations % 10 == 0 and print_intermediate:
            print(f"Iteration number {i}\n Current loss {loss.item()}\n")

    results_dict = {
        "opt_params": {
            "alpha": opt.param_groups[0]["params"][0],
            "mu": opt.param_groups[0]["params"][1],
            "sigma": opt.param_groups[0]["params"][2],
        },
        "loss_history": loss_history,
    }
    return results_dict


# In[139]:


def training_workflow(
    U: callable,
    W: callable,
    D: callable,
    n_qubits: int,
    k: int,
    probs_func: callable,
    W_layers: int,
    gamma_length: int,
    n_samples: int,
    p: int,
    num_series: int,
    noise_amp: float,
    t_init: float,
    t_end: float,
    batch_size: int,
    tau: float,
    pytorch_optimizer: callable,
    lr: float,
    batch_iterations: int,
):
    """Combine all of the previously defined functions for an entire training workflow."""

    X, T = generate_normal_time_series_set(p, num_series, noise_amp, t_init, t_end)
    Xtr = make_atomized_training_set(X, T)
    cycler = get_training_cycler(Xtr, batch_size)
    init_params = get_initial_parameters(W, W_layers, n_qubits)
    callable_penalty = callable_arctan_penalty(tau)
    callable_proj = get_callable_projector_func(k, U, W, D, n_qubits, probs_func)
    #xt, t = X[0].unsqueeze(0), T[0].unsqueeze(0)
    #gamma, alpha = 1, 5
    #qml.draw_mpl(get_probs, decimals=2)(xt, t, alpha, gamma, k, U, W, D, n_qubits)
    results_dict = train_model_gradients(
        lr,
        init_params,
        pytorch_optimizer,
        cycler,
        n_samples,
        callable_penalty,
        batch_iterations,
        callable_proj,
        gamma_length,
        print_intermediate=False,
    )
    return results_dict


# In[140]:


general_options = {
    "U": qml.AngleEmbedding,
    "W": qml.StronglyEntanglingLayers,
    "D": D,
    "n_qubits": 1,
    "probs_func": get_probs,
    "gamma_length": 1,
    "n_samples": 10,
    "p": 25,
    "num_series": 25,
    "noise_amp": 0.1,
    "t_init": 0.1,
    "t_end": 2 * torch.pi,
    "k": 1,
}

training_options = {
    "batch_size": 10,
    "tau": 5,
    "pytorch_optimizer": torch.optim.Adam,
    "lr": 0.01,
    "batch_iterations": 100,
    "W_layers": 3,
}

training_options.update(general_options)


# In[141]:


results_dict = training_workflow(training_options["U"],
    training_options["W"],
    training_options["D"],
    training_options["n_qubits"],
    training_options["k"],
    training_options["probs_func"],
    training_options["W_layers"],
    training_options["gamma_length"],
    training_options["n_samples"],
    training_options["p"],
    training_options["num_series"],
    training_options["noise_amp"],
    training_options["t_init"],
    training_options["t_end"],
    training_options["batch_size"],
    training_options["tau"],
    training_options["pytorch_optimizer"],
    training_options["lr"],
    training_options["batch_iterations"]
)
results_dict


# In[108]:


plt.figure()
plt.plot(results_dict["loss_history"], ".-")
plt.ylabel("Loss [$\mathcal{L}$]")
plt.xlabel("Batch iterations")
plt.title("Loss function versus batch iterations in training")
plt.grid()


# In[109]:


def get_preds_given_threshold(zeta: float, scores: torch.Tensor) -> torch.Tensor:
    """For a given threshold, get the predicted labels (1 or -1), given the anomaly scores."""
    return torch.tensor([-1 if score > zeta else 1 for score in scores])

def get_truth_labels(
    normal_series_set: torch.Tensor, anomalous_series_set: torch.Tensor
) -> torch.Tensor:
    """Get a 1D tensor containing the truth values (1 or -1) for a given set of
    time series.
    """
    norm = torch.ones(normal_series_set.size()[0])
    anom = -torch.ones(anomalous_series_set.size()[0])
    return torch.cat([norm, anom])

def get_accuracy_score(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Given the predictions and truth values, return a number between 0 and 1
    indicating the accuracy of predictions.
    """
    return torch.sum(pred == truth) / truth.size()[0]


# In[112]:


def threshold_scan_acc_score(
    scores: torch.Tensor, truth_labels: torch.Tensor, zeta_min: float, zeta_max: float, steps: int
) -> torch.Tensor:
    """Given the anomaly scores and truth values,
    scan over a range of thresholds = [zeta_min, zeta_max] with a
    fixed number of steps, calculating the accuracy score at each point.
    """
    accs = torch.empty(steps)
    for i, zeta in enumerate(torch.linspace(zeta_min, zeta_max, steps)):
        preds = get_preds_given_threshold(zeta, scores)
        accs[i] = get_accuracy_score(preds, truth_labels)
    return accs

def get_anomaly_score(
    callable_proj: callable,
    y: torch.Tensor,
    T: torch.Tensor,
    alpha_star: torch.Tensor,
    mu_star: torch.Tensor,
    sigma_star: torch.Tensor,
    gamma_length: int,
    n_samples: int,
    get_time_resolved: bool = False,
):
    """Get the anomaly score for an input time series y. """
    scores = torch.empty(T.size()[0])
    for i in range(T.size()[0]):
        scores[i] = (
            1
            - F(
                callable_proj,
                y[i].unsqueeze(0),
                T[i].unsqueeze(0),
                alpha_star,
                mu_star,
                sigma_star,
                gamma_length,
                n_samples,
            )
        ).square()
    if get_time_resolved:
        return scores, scores.mean()
    else:
        return scores.mean()

def get_norm_and_anom_scores(
    X_norm: torch.Tensor,
    X_anom: torch.Tensor,
    T: torch.Tensor,
    callable_proj: callable,
    model_params: dict,
    gamma_length: int,
    n_samples: int,
) -> torch.Tensor:
    """Get the anomaly scores assigned to input normal and anomalous time series instances."""
    alpha = model_params["alpha"]
    mu = model_params["mu"]
    sigma = model_params["sigma"]
    norm_scores = torch.tensor(
        [
            get_anomaly_score(callable_proj, xt, T, alpha, mu, sigma, gamma_length, n_samples)
            for xt in X_norm
        ]
    )
    anom_scores = torch.tensor(
        [
            get_anomaly_score(callable_proj, xt, T, alpha, mu, sigma, gamma_length, n_samples)
            for xt in X_anom
        ]
    )
    return torch.cat([norm_scores, anom_scores])


# In[114]:


def threshold_tuning_workflow(
    opt_params: dict,
    gamma_length: int,
    n_samples: int,
    probs_func: callable,
    zeta_min: float,
    zeta_max: float,
    steps: int,
    p: int,
    num_series: int,
    noise_amp: float,
    spike_amp: float,
    max_duration: int,
    t_init: float,
    t_end: float,
    k: int,
    U: callable,
    W: callable,
    D: callable,
    n_qubits: int,
    random_model_seeds: torch.Tensor,
    W_layers: int,
) -> tuple:
    """A workflow for tuning the threshold value zeta, in order to maximize the accuracy score
    for a validation data set. Results are tested against random models at their optimal zetas.
    """
    # Generate datasets
    X_val_norm, T = generate_normal_time_series_set(p, num_series, noise_amp, t_init, t_end)
    X_val_anom, T = generate_anomalous_time_series_set(
        p, num_series, noise_amp, spike_amp, max_duration, t_init, t_end
    )
    truth_labels = get_truth_labels(X_val_norm, X_val_anom)

    # Initialize quantum functions
    callable_proj = get_callable_projector_func(k, U, W, D, n_qubits, probs_func)

    accs_list = []
    scores_list = []
    # Evaluate optimal model
    scores = get_norm_and_anom_scores(
        X_val_norm, X_val_anom, T, callable_proj, opt_params, gamma_length, n_samples
    )
    accs_opt = threshold_scan_acc_score(scores, truth_labels, zeta_min, zeta_max, steps)
    accs_list.append(accs_opt)
    scores_list.append(scores)

    # Evaluate random models
    for seed in random_model_seeds:
        rand_params = get_initial_parameters(W, W_layers, n_qubits, seed)
        scores = get_norm_and_anom_scores(
            X_val_norm, X_val_anom, T, callable_proj, rand_params, gamma_length, n_samples
        )
        accs_list.append(threshold_scan_acc_score(scores, truth_labels, zeta_min, zeta_max, steps))
        scores_list.append(scores)
    return accs_list, scores_list


# In[113]:


threshold_tuning_options = {
    "spike_amp": 0.4,
    "max_duration": 5,
    "zeta_min": 0,
    "zeta_max": 1,
    "steps": 100000,
    "random_model_seeds": [0, 1],
    "W_layers": 2,
    "opt_params": results_dict["opt_params"],
}

threshold_tuning_options.update(general_options)


# In[117]:


ct_val_results = threshold_tuning_workflow(
    threshold_tuning_options["opt_params"],
    threshold_tuning_options["gamma_length"],
    threshold_tuning_options["n_samples"],
    threshold_tuning_options["probs_func"],
    threshold_tuning_options["zeta_min"],
    threshold_tuning_options["zeta_max"],
    threshold_tuning_options["steps"],
    threshold_tuning_options["p"],
    threshold_tuning_options["num_series"],
    threshold_tuning_options["noise_amp"],
    threshold_tuning_options["spike_amp"],
    threshold_tuning_options["max_duration"],
    threshold_tuning_options["t_init"],
    threshold_tuning_options["t_end"],
    threshold_tuning_options["k"],
    threshold_tuning_options["U"],
    threshold_tuning_options["W"],
    threshold_tuning_options["D"],
    threshold_tuning_options["n_qubits"],
    threshold_tuning_options["random_model_seeds"],
    threshold_tuning_options["W_layers"],
)
ct_val_results


# In[120]:


accs_list, scores_list = ct_val_results


# In[153]:


zeta_xlims = [(0, 0.001), (0.12, 0.38), (0.25, 0.38)]
titles = ["Trained model", "Random model 1", "Random model 2"]
zetas = torch.linspace(
    threshold_tuning_options["zeta_min"],
    threshold_tuning_options["zeta_max"],
    threshold_tuning_options["steps"],
)

fig, axs = plt.subplots(ncols=3, nrows=2, sharey="row")
fig.set_figheight(8)
fig.set_figwidth(12)

for i in range(3):
    axs[0, i].plot(zetas, accs_list[i])
    axs[0, i].set_xlim(zeta_xlims[i])
    axs[0, i].set_xlabel("Threshold [$\zeta$]")
    axs[0, i].set_title(titles[i])
    axs[1, i].boxplot(
        [
            scores_list[i][0 : general_options["num_series"]],
            scores_list[i][general_options["num_series"] : -1],
        ],
        labels=["Normal", "Anomalous"],
    )
    axs[1, i].set_yscale("log")
    axs[1, i].axhline(
        zetas[torch.argmax(accs_list[i])], color="k", linestyle=":", label="Optimal $\zeta$"
    )
    axs[1, i].legend()
axs[0, 0].set_ylabel("Accuracy score")
axs[1, 0].set_ylabel("Anomaly score")
fig.tight_layout()
plt.savefig("qmodel.pdf", dpi=300)


# In[ ]:




