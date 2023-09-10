import React, { useEffect } from 'react';
import cls from './Modal.module.scss';
import Portal from '../Portal/Portal';
import { classNames } from '../../helpers/ClassNames/classNames';

const Modal = (props) => {
  const {
    className,
    children,
    isOpen
  } = props;

  const mods = {
    [cls.opened]: isOpen
  }

  return (
    <Portal>
      <div className={classNames(cls.overlay, mods, [])}></div>
      <div className={classNames(cls.Modal, mods, [className])}>
        <div className={cls.content}>
          {children}
        </div>
      </div>
    </Portal>
  )
}

export default Modal;