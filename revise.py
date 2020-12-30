from collections import namedtuple
from functools import partial

from jax import grad, vmap, jit
import jax.numpy as np
from jax.scipy.special import xlogy
import matplotlib.pyplot as plt
import numpy
#from scipy.stats import norm
from jax.scipy.stats import norm
from tqdm.auto import tqdm

Path = namedtuple('Path', ['x', 'z', 'grad_z', 'prob_target', 'prob_data', 'details'], defaults=[[],[],[],[]])
LOSS_BLEND_PARAM = 2e-3

class Revise:
    def __init__(self, classifier, vae, calc_loss=None, target_class=1):
        self.classifier = classifier
        self.vae = vae
        if calc_loss is None:
            calc_loss = lambda t, p: _binary_crossentropy(t, p).mean()
        self._calc_loss = calc_loss
        self._target_class = target_class
        

    def _blended_shortest_path_to_target_class(
        self, chosen_point, learning_rate=1e-3, dist_weight=1e-5, 
        max_iter=200, min_prob_target=0.5, min_prob_data=0.9, calc_dist=None):
        
        if calc_dist == None:
            calc_dist = _euclidean
            
        # Make it just a function of z
        simple_objective = partial(_blended_revise_objective,
                                   chosen_point=chosen_point,
                                   vae=self.vae,
                                   classifier=self.classifier,
                                   calc_loss=self._calc_loss,
                                   dist_weight=dist_weight,
                                   calc_dist=calc_dist,
                                   target_class=self._target_class)
        
        grad_objective_wrt_z = vmap(grad(simple_objective, has_aux=True))
        
        new_z = self.vae.encode(chosen_point)[:,:self.vae.n_latent_dims]
        path = Path(
            x=[chosen_point],
            z=[new_z],
            grad_z=[],
            prob_target=[],
            prob_data=[],
            details=[])
        
        pbar = tqdm(total=max_iter, smoothing=0)
        i = 0
        reconstructed_prob = 0
    
        while i < max_iter and (np.array(reconstructed_prob < min_prob_target).any()
                                or np.array(prob_data < min_prob_data).any()):
            grad_z, obj_info = grad_objective_wrt_z(new_z)
            new_z -= learning_rate * grad_z
            path = update_path(path, grad_z, new_z, obj_info)
            pbar.update()
            i += 1

        pbar.close()
        n_paths = chosen_point.shape[0]
        return _split_paths(path, n_paths, min_prob_target)

        
    def _base_shortest_path_to_target_class(
        self, chosen_point, learning_rate=1e-3, dist_weight=1e-5, 
        max_iter=200, min_prob_target=0.5, calc_dist=None):
        
        if calc_dist == None:
            calc_dist = _euclidean
            
        # Make it just a function of z
        simple_objective = partial(_base_revise_objective,
                                   chosen_point=chosen_point,
                                   vae=self.vae,
                                   classifier=self.classifier,
                                   calc_loss=self._calc_loss,
                                   dist_weight=dist_weight,
                                   calc_dist=calc_dist,
                                   target_class=self._target_class)
        
        grad_objective_wrt_z = vmap(grad(simple_objective, has_aux=True))
        
        new_z = self.vae.encode(chosen_point)[:,:self.vae.n_latent_dims]
        path = Path(
            x=[chosen_point],
            z=[new_z],
            grad_z=[],
            prob_target=[],
            prob_data=[],
            details=[])
        
        pbar = tqdm(total=max_iter, smoothing=0)
        i = 0
        reconstructed_prob = 0
    
        while i < max_iter and np.array(reconstructed_prob < min_prob_target).any():
            grad_z, obj_info = grad_objective_wrt_z(new_z)
            new_z -= learning_rate * grad_z
            path = update_path(path, grad_z, new_z, obj_info)
            pbar.update()
            i += 1

        pbar.close()
        n_paths = chosen_point.shape[0]
        return _split_paths(path, n_paths, min_prob_target)
        
        
    def shortest_path_to_target_class(self, *args, **kwargs):
        return self._base_shortest_path_to_target_class(*args, **kwargs)
    
        
    def show_path(self, path, dataset=None, zoom=False, landscape='loss', ax=None, fig=None):
        '''Assumes two-dimensional space'''
        landscape_fns = {
            'loss': self._base_loss_landscape,
            'blended_loss': self._blended_loss_landscape,
            'likelihood': self._likelihood_landscape,
            'prob_target': self._prob_target_landscape
        }
        grid, positions = self._grid_positions(dataset)
        x, y = grid
        loss = landscape_fns[landscape](positions).reshape(grid.shape[1:3])
        titles = {
            'loss': 'REVISE path against\n REVISE objective',
            'blended_loss': 'REVISE path against\n REVISE objective',
            'likelihood': 'REVISE path against\n data log-likelihood under VAE',
            'prob_target': 'REVISE path against\n classifier probability of target class'
        }
        title = titles[landscape]
        chosen_point = path.x[0]
        if ax is None:
            fig, ax = plt.subplots(1,1)
        contour = ax.contourf(x, y, loss, levels=100 if zoom else 50, cmap='viridis')
        fig.colorbar(contour, ax=ax)
        if dataset is not None:
            ax.scatter(dataset.T[0], dataset.T[1], color='white', alpha=0.6)
        ax.scatter(chosen_point[0], chosen_point[1], color='red')
        x_path = np.array(path.x).squeeze()
        if zoom:
            ax.set_xlim(x_path[:,0].min() - .1, x_path[:,0].max() + .1)
            ax.set_ylim(x_path[:,1].min() - .1, x_path[:,1].max() + .1)
            title += ' (zoomed)'
        #iter_by = (x_path.shape[0] // 10) + 1
        #plt.plot(x_path[::iter_by,0], x_path[::iter_by,1], c='red', marker='o', alpha=0.7)
        ax.set_title(title)
        ax.plot(x_path[:,0], x_path[:,1], c='red', alpha=0.5, marker='.')
        return ax

        
    def _base_loss_landscape(self, positions):
        '''Assumes two-dimensional space'''
        prob_target = self.classifier.predict(positions)
        target = numpy.ones_like(prob_target) * self._target_class
        return _binary_crossentropy(target, prob_target).squeeze()
    
    
    def _blended_loss_landscape(self, positions):
        '''Assumes two-dimensional space'''
        prob_target = self.classifier.predict(positions)
        target = numpy.ones_like(prob_target) * self._target_class
        z = self.vae.encode(positions)[:,:self.vae.n_latent_dims]
        reconstructed_x = self.vae.decode(z)
        log_likelihood_data = (
            norm.logpdf(positions, reconstructed_x, self.vae.x_var ** 0.5)).sum(axis=1)
        likelihood_data_term = LOSS_BLEND_PARAM * (log_likelihood_data + np.log(prob_target.squeeze()))
        return _binary_crossentropy(target, prob_target).squeeze() - likelihood_data_term
    
    
    def _likelihood_landscape(self, positions):
        z = self.vae.encode(positions)
        new_x = self.vae.decode(z[:,:2])
        log_prob = numpy.sum(norm.logpdf(positions, new_x, self.vae.x_var ** 0.5), axis=1)
        return log_prob
    
    
    def _prob_target_landscape(self, positions):
        prob_target = self.classifier.predict(positions)
        return prob_target
    
    
    def _grid_positions(self, dataset=None, grid=None):
        if grid is None and dataset is None:
            grid = numpy.mgrid[-5:5:0.2,-5:5:0.2]
        if dataset is not None:
            x_lim = (dataset[:,0].min(), dataset[:,0].max())
            y_lim = (dataset[:,1].min(), dataset[:,1].max())
            x_margin = (x_lim[1]-x_lim[0]) * 0.2
            y_margin = (y_lim[1]-y_lim[0]) * 0.2
            grid = numpy.mgrid[
                x_lim[0]-x_margin:x_lim[1]+x_margin:(x_lim[1]-x_lim[0])/100,
                y_lim[0]-y_margin:y_lim[1]+y_margin:(y_lim[1]-y_lim[0])/100]
        x, y = grid
        return grid, numpy.vstack((x.flatten(), y.flatten())).T
    
    
    def _calc_loss_at_point(self, point, target):
        prob_target = self.classifier.predict(numpy.array(point))
        return self._calc_loss(target, prob_target)


def _blended_revise_objective(latent_pos, chosen_point,
                      vae, classifier,
                      calc_loss,
                      dist_weight, calc_dist, target_class):
    reconstructed_x = vae.decode(latent_pos)
    
    distance = calc_dist(chosen_point, reconstructed_x)
    distance_term = dist_weight * distance
    
    reconstructed_prob_target = classifier.predict(reconstructed_x)
    target_loss = calc_loss(np.array([1]), reconstructed_prob_target)
    
    reconstructed_z = vae.encode(reconstructed_x)[:vae.n_latent_dims]
    rereconstructed_x = vae.decode(reconstructed_z)
    log_likelihood_data = (norm.logpdf(reconstructed_x, rereconstructed_x, vae.x_var ** 0.5)).sum()
    likelihood_data_term = LOSS_BLEND_PARAM * (log_likelihood_data + np.log(reconstructed_prob_target.squeeze()))
    
    objective = target_loss + distance_term - likelihood_data_term
    return objective.squeeze(), {
        'reconstructed_x': reconstructed_x,
        'reconstructed_prob_target': reconstructed_prob_target,
        'log_likelihood_data': log_likelihood_data
    }


def _base_revise_objective(latent_pos, chosen_point,
                      vae, classifier,
                      calc_loss,
                      dist_weight, calc_dist, target_class):
    
    reconstructed_x = vae.decode(latent_pos)
    
    distance = calc_dist(chosen_point, reconstructed_x)
    distance_term = dist_weight * distance
    
    reconstructed_prob_target = classifier.predict(reconstructed_x)
    target_loss = calc_loss(np.array([target_class]), reconstructed_prob_target)
    
    objective = target_loss + distance_term
    return objective.squeeze(), {
        'reconstructed_x': reconstructed_x,
        'reconstructed_prob_target': reconstructed_prob_target
    }


@jit
def _euclidean(x1, x2):
    return np.linalg.norm(x1 - x2, ord=2)


@jit
def _binary_crossentropy(true, prob):
    return -(xlogy(true, prob) + xlogy(1 - true, 1 - prob))


def _split_paths(path, n_paths, min_prob_of_target):
    paths = []
    x = np.array(path.x)
    z = np.array(path.z)
    grad_z = np.array(path.grad_z)
    prob_target = np.array(path.prob_target)
    prob_data = np.array(path.prob_data)
    for i in range(n_paths):
        paths.append(Path(
            x[:,i,:].squeeze(),
            z[:,i,:].squeeze(),
            grad_z[:,i,:].squeeze(),
            prob_target[:,i,:].squeeze(),
            prob_data[:,i].squeeze()))
    return paths


def trim_path(path, min_prob_target=-1, min_prob_data=-1):
    assert min_prob_target > 0 or min_prob_data > 0
    i_to_keep = (path.prob_target < min_prob_target) | (path.prob_data < min_prob_data)
    return Path(
        x=path.x[i_to_keep],
        z=path.z[i_to_keep],
        grad_z=path.grad_z[i_to_keep],
        prob_target=path.prob_target[i_to_keep],
        prob_data=path.prob_data[i_to_keep],
    )


def update_path(path, grad_z, new_z, obj_info):
    path.x.append(obj_info['reconstructed_x'])
    path.z.append(new_z)
    path.grad_z.append(grad_z)
    path.prob_target.append(obj_info['reconstructed_prob_target'])
    path.prob_data.append(obj_info.get('log_likelihood_data', np.zeros_like(obj_info['reconstructed_prob_target']) - 1))
    return path