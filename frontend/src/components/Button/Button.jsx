import React from 'react';
import cls from './Button.module.scss';
import { classNames } from '../../helpers/ClassNames/classNames';

const Button = (props) => {

  const {
    className,
    children,
    disabled,
    ...otherProps
  } = props;

  const mods = {
    [cls['disabled']]: disabled
  }

  return (
    <button
      className={classNames(cls.Button, mods, [className])}
      {...otherProps}
      disabled={disabled}
    >
      {children}
    </button>
  )
}

export default Button;