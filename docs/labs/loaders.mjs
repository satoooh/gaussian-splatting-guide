export const loaders = {
  real3d: () => import('./real3d.mjs'),
  training: () => import('./training.mjs'),
  covariance: () => import('./covariance.mjs'),
  sh: () => import('./sh.mjs'),
  projection: () => import('./projection.mjs'),
  compositing: () => import('./compositing.mjs'),
  splat3d: () => import('./splat3d.mjs'),
  interp: () => import('./interp.mjs'),
  spacetime: () => import('./spacetime.mjs'),
};

export async function mountLab(id, root) {
  const { mount } = await loaders[id]();
  root.classList.add('lab-widget');
  mount(root);
}
