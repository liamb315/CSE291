function target = getImageLabels(filename, dataset)
%String parser that takes the filename and extracts the emotion and
%identity that corresponds to the image

if dataset == 1 %NIMSTIM
    temp=regexp(filename,'[_-]','split');
    target = str2num(temp{1}(1:2)) - 22;
    if target == 23
        target = 22;
    end
        
    elseif dataset == 2 %POFA
        temp=regexp(filename,'[_-]','split');
        if temp{2} == 'AN'
            target = 1;
        elseif temp{2} == 'DI'
            target = 2;
        elseif temp{2} == 'HA'
            target = 3;
        elseif temp{2} == 'SP'
            target = 4;
        elseif temp{2} == 'SA'
            target = 5;
        elseif temp{2} == 'FE'
            target = 6;
        end
    end

end