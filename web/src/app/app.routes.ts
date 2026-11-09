import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    title: 'Inicio · CardioRed',
    loadComponent: () => import('./pages/dashboard/dashboard').then((m) => m.Dashboard),
  },
  {
    path: 'prediccion',
    title: 'Predicción · CardioRed',
    loadComponent: () => import('./pages/prediction/prediction').then((m) => m.Prediction),
  },
  {
    path: 'modelo',
    title: 'Modelo y datos · CardioRed',
    loadComponent: () => import('./pages/model-info/model-info').then((m) => m.ModelInfo),
  },
  { path: '**', redirectTo: '' },
];
