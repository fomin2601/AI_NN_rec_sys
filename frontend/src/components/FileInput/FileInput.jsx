import React from 'react';
import cls from './FileInput.module.scss';
import Button from '../Button/Button';
import { classNames } from '../../helpers/ClassNames/classNames';

const FileInput = (props) => {
  const {
    name,
    changeHandler,
    file,
    accept,
    title,
    disabled
  } = props;

  const mods = {
    [cls.disabled]: disabled
  }

  return (
    <div>
    <label className={classNames(cls.label, mods, [])}>
      <div className={cls.custom}>
        {title}
      </div>
      <input
        name={name}
        accept={accept}
        multiple={false}
        className={cls.input}
        type='file'
        onChange={changeHandler}
      />
    </label>
    </div>
  )
}

export default FileInput;