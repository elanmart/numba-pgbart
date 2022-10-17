from typing import Any

import numpy as np
from numba import float64 as f64
from numba import int64 as i64
from numba import uint8 as u8
from numba.core.types import DictType, ListType, Tuple, FunctionType
from numba.experimental import jitclass
from numba.typed.typedlist import List

from .tree import Node, Tree
from .particle import Particle, ParticleParams

NOTHING = u8(0)


def NumbaType(jit_cls) -> Any:
    return jit_cls.class_type.instance_type


@jitclass(
    spec=[
        ("n_trees", i64),
        ("n_particles", i64),
        ("alpha", f64),
        ("default_kf", f64),
        ("batch", Tuple((f64, f64))),
        ("intial_alpha_vec", f64[:]),
    ]
)
class PgBartSettings:
    def __init__(
        self,
        n_trees,
        n_particles,
        alpha,
        default_kf,
        batch,
        intial_alpha_vec,
    ):
        self.n_trees = n_trees
        self.n_particles = n_particles
        self.alpha = alpha
        self.default_kf = default_kf
        self.batch = batch
        self.intial_alpha_vec = intial_alpha_vec


@jitclass(
    spec=[
        ("alpha_vec", f64[:]),
        ("spliting_probs", f64[:]),
        ("alpha", f64),
    ]
)
class Probabilities:
    def __init__(self):
        pass

    def normal(self):
        return np.random.randn() * self.sigma + self.mu

    def uniform(self):
        return np.random.uniform(self.low, self.high)

    def sample_expand_flag(self, node):
        d = node.depth()
        p = 1.0 - np.power(self.alpha, d)
        res = p < np.random.random()

        return res

    def sample_leaf_value(self, mu, kfactor):
        norm = self.normal() * kfactor
        return norm + mu


    def sample_split_index(self):
        split_probs = self.probabilities.spliting_probs
        rnd = np.random.random()

        for (idx, value) in enumerate(split_probs):
            if rnd <= value:
                return idx

        return len(split_probs) - 1

    def sample_split_value(self, candidates):
        return np.random.choice(candidates)

    def sample_kf(self):
        return self.uniform()

    def select_particle(self, particles, weights):
        idx = np.random.choice(np.arange(len(weights)), p=weights)
        return particles[idx]

    def resample_particles(self, particles, weights):
        inds = np.random.choice(np.arange(len(weights)), len(weights), p=weights) + 2
        ret = List([particles[0].copy(), particles[1].copy()])

        for idx in inds:
            ret.append(particles[idx].copy())

        return ret


@jitclass(
    spec=[
        ("X", f64[:, :]),
        ("y", f64[:]),
        ("logp", FunctionType(f64(f64[:]))),
    ]
)
class ExternalData:
    def __init__(self):
        pass


@jitclass(
    spec=[
        ("data", NumbaType(ExternalData)),
        ("params", NumbaType(PgBartSettings)),
        ("probabilities", NumbaType(Probabilities)),
        ("particles", ListType(NumbaType(Particle))),
        ("predictions", f64[:]),
        ("variable_inclusion", i64[:]),
        ("tune", bool),
    ]
)
class PgBartState:
    def __init__(self, params, data):
        self.params = params
        self.data = data

        m = float(params.n_trees)
        mu = np.mean(data.y)
        leaf_value = mu / m

        binary = np.all(
            (data.y == 0.)
            | (data.y == 1.)
        )

        std = 0 / np.float_power(m, 0.5) if binary else np.std(data.y) / np.float_power(m, 0.5)
        predictions = np.full((data.X.shape[0], ), mu)
        variable_inclusion = np.zeros((data.X.shape[1], ), dtype=np.int64)
        
        particles = List.empty_list(NumbaType(Particle))
        for _ in range(params.n_trees):
            p_params = ParticleParams().init(data.X.shape[0], data.X.shape[1], params.kfactor)
            particle = Particle().init(p_params, leaf_value)
            particles.append(particle)

        alpha_vec = params.initial_alpha_vec.copy()
        splitting_probs = normalzied_cumsum(alpha_vec)
        probabilites = Probabilities()

    def step(self):

        amount = self.num_to_update()
        lengths = np.arange(self.params.n_trees)
        indices = np.random.choice(length, amount, replace=False)

        for particle_idx in indices:
            
            # Grow
            selected_p = self.particles[particle_idx]
            local_preds = self.predictions - selected_p.predict()
            local_particles = self.initialize_particles(selected_p, local_preds, mu)
            local_particles = self.grow_particles(local_particles, local_preds)
            _, weights = self.normalize_weights(local_particles)
            
            # Select
            ix = np.random.choice(np.arange(len(weights)), 1, p=weights)
            selected = local_particles[ix]
            log_n_particles = np.log(self.params.n_particles)
            log_lik = selected.weight.log_likelihood
            selected.weight.log_w = log_lik - log_n_particles

            # Update self
            self.predictions = local_preds + selected.predict()
            self.particles[particle_idx] = selected
            self.update_sampling_preds(selected)

    def initialize_particles(self, p, local_preds, mu):
        p0 = p.frozen_copy()
        p1 = p.with_resampled_leaves(self)
        local_particles = List.empty_list(NumbaType(Particle))
        local_particles.append(p0)
        local_particles.append(p1)

        for item in local_particles:
            preds = local_preds + item.predict()
            log_lik = self.data.model_logp(preds)
            item.weight.reset(log_lik)

        for _ in range(2, self.params.n_particles):

            if self.tune:
                kf = self.probabilities.sample_kf()
                params = p.params.with_new_kf(kf)
            else:
                params = p.params.copy()

            new_p = Particle().init(params, mu)
            local_particles.append(new_p)

        return local_particles

    def all_done(self, particles):
        done = True
        for p in particles:
            if not p.finished():
                done = False
                break

        return done

    def grow_particles(self, particles, local_preds):
        X =     self.data.X
        while True:

            if self.all_done(particles):
                break
            
            for p in particles[2:]:
                needs_update = p.grow(X, self)
                if needs_update:
                    preds = local_preds + p.predict()
                    loglik = self.data.model_lopgp(preds)
                    p.weight.update(loglik)

            wt, weights = self.normalize_weights(particles[2:])
            particles = self.probabilites.resample_particles(particles, weights)
            for p in particles[2:]:
                p.weight.log_w = wt

        for p in partciles:
            loglik = p.weight.log_likelihood()
            p.weight.log_w = loglik

        return particles

    def update_sampling_probs(self, p):
        used_variates = p.split_variables()

        if self.tune:
            
            for idx in used_variates:
                self.probbabilies.alpha_vec[idx] += 1

            probs = normalized_cumsum(self.probabilities.alpha_vec)
            self.splitting_probs = probs

        else:
            for idx in used_variates:
                self.variable_inclusion[idx] += 1

    def normalize_weights(particles):
        log_w = np.empty((len(particles), ), dtype=np.float64)
        for i in range(len(particles)):
            log_w[i] = particles[i].weight.log_w

        max_log_w = np.max(log_w)
        scaled_low_w = log_w - max_log_w
        scaled_w = np.exp(scaled_log_w)
        scaled_w_sum = np.sum(scaled_w)
        log_scaled_w_sum = np.log(scaled_w_sum)
        normalized_scaled_w = (scaled_w / scaled_w_sum) + 1e-12
        log_n_particles = np.log(self.params.n_particles)
        log_w = max_log_w + log_scaled_w_sum - log_n_particles

        return log_w, normalized_scaled_w

    def num_to_update(self):
        if self.tune:
            frac = self.params.batch[0]
        else:
            frac = self.params.batch[1]

        return int(np.floor(self.params.n_trees * fraction))
